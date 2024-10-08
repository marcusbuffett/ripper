#![feature(type_alias_impl_trait)]
use rule::Rule;
use tracing::{debug, warn};

use rand;
use rand::Rng;
use std::collections::HashMap;

mod antd;
mod attribute;
mod attribute_info;
mod instance;
mod rule;
mod rule_set;
mod rule_stats;

use antd::NominalAntd;
use attribute::Attribute;
use instance::{Instance, Instances};
use rule_set::RuleSet;
use rule_stats::RuleStats;

pub struct Rip {
    class: Option<String>,
    ruleset: Vec<Rule>,
    distributions: Vec<Vec<f64>>,
    optimizations: usize,
    folds: usize,
    min_no: f64,
    debug: bool,
    check_err: bool,
    use_pruning: bool,
    seed: u64,
    total: f64,
    ruleset_stats: Vec<RuleStats>,
}

const MAX_DL_SURPLUS: f64 = 64.0;

impl Rip {
    pub fn new() -> Self {
        Rip {
            class: None,
            ruleset: Vec::new(),
            distributions: Vec::new(),
            optimizations: 2,
            folds: 3,
            min_no: 2.0,
            debug: false,
            check_err: true,
            use_pruning: true,
            seed: 1,
            total: 0.0,
            ruleset_stats: Vec::new(),
        }
    }

    pub fn build_classifier(&mut self, instances: &mut Instances) -> Result<(), String> {
        // Remove instances with missing class
        // don't need to do this, as they will all have a class index
        // instances.delete_with_missing_class();

        let mut rng = rand::thread_rng();

        self.total = RuleStats::num_all_conditions(instances) as f64;
        debug!("Number of all possible conditions = {}", self.total);

        // Sort instances by class frequency
        let mut data = instances.clone();
        data.sort_by_class_freq_asc();

        self.ruleset = vec![];
        self.ruleset_stats = vec![];
        self.distributions = vec![];

        let class_frequencies = data.class_frequencies();
        for (class_index, (class, count)) in class_frequencies.iter().enumerate() {
            let total_remaining_instances = class_frequencies
                .iter()
                .skip(class_index)
                .map(|(_, c)| *c)
                .sum::<usize>();
            let expected_fp_rate = *count as f64 / total_remaining_instances as f64;
            let total_weight = data.instances.len();
            let class_weight = *count as f64;

            // DL of default rule
            let default_dl = RuleStats::data_description_length(
                expected_fp_rate,
                0.0,
                total_weight as f64,
                0.0,
                class_weight as f64,
            );
            debug!("Default DL: {}", default_dl);
            let (rules, stats, data) =
                self.ruleset_for_one_class(&data, expected_fp_rate, class_index, default_dl);
        }

        Ok(())
    }

    pub fn distribution_for_instance(&self, instance: &Instance) -> Vec<f64> {
        // Implementation of distributionForInstance goes here
        unimplemented!()
    }

    pub fn ruleset_for_one_class(
        &self,
        data: &Instances,
        expected_fp_rate: f64,
        class_index: usize,
        default_description_length: f64,
    ) -> (Vec<Rule>, RuleStats, Instances) {
        debug!("Generating a ruleset to predict {}", class_index);
        // let mut stop = false;
        let mut ruleset: Vec<Rule> = vec![];

        let mut dl = default_description_length;
        let mut min_dl = default_description_length;

        let mut stats: Option<RuleStats> = None;
        let mut rst = vec![];
        let mut new_data: Instances = data.clone();
        let mut last_grow_data: Option<Instances> = None;
        let mut last_prune_data: Option<Instances> = None;

        let mut has_positive = true;

        loop {
            let mut new_data = RuleStats::stratify(&new_data, self.folds);
            let (mut grow_data, prune_data) = Instances::partition(&new_data, self.folds);
            dbg!(
                &new_data.instances.len(),
                &grow_data.instances.len(),
                &prune_data.instances.len()
            );
            last_grow_data = Some(grow_data.clone());
            last_prune_data = Some(prune_data.clone());
            let mut one_rule = Rule::new();
            one_rule.consequent = class_index.clone();
            debug!(
                "Growing a rule with {} instances",
                grow_data.instances.len()
            );
            one_rule.grow(&mut grow_data).expect("Grow failed");
            debug!("Rule before pruning: {:?}", one_rule);
            debug!("Pruning the rule");
            one_rule.prune(&prune_data, false);
            debug!("Rule after pruning: {:?}", one_rule);
            let mut stats = match stats {
                None => {
                    let mut stats = (RuleStats::new());
                    stats.total = self.total;
                    stats.data = Some(new_data.clone());
                    stats
                }
                Some(ref stats) => stats.clone(),
            };
            stats.add_and_update(&one_rule);
            let last_rule_index = stats.get_ruleset_size() - 1;
            dl += stats.relative_dl(last_rule_index, expected_fp_rate);
            min_dl = min_dl.min(dl);
            rst = stats.get_simple_stats(last_rule_index).clone();
            debug!("The rule covers: {:?} | pos = {:?} | neg = {:?}\nThe rule doesn't cover: {:?} | pos = {:?}", 
                rst[0], rst[2], rst[4], rst[1], rst[5]);
            let stop = check_stop(&rst, min_dl, dl);
            if stop {
                stats.remove_last();
                break;
            }
            ruleset.push(one_rule);
            new_data = stats.get_filtered(last_rule_index).1.clone();
            let has_positive = rst[5] > 0.0;
            if !has_positive {
                break;
            }
        }

        warn!("TODO: optimization stage");

        debug!("Final ruleset: {:?}", ruleset);

        (
            ruleset.clone(),
            stats.clone().unwrap(),
            if ruleset.len() > 0 {
                stats.unwrap().get_filtered(ruleset.len() - 1).1.clone()
            } else {
                data.clone()
            },
        )
    }

    // Other methods...
}

fn check_stop(rst: &[f64], min_dl: f64, dl: f64) -> bool {
    if dl > min_dl + MAX_DL_SURPLUS {
        debug!("DL too large: {} | {}", dl, min_dl);
        true
    } else if rst[2] <= 0.0 {
        debug!("Too few positives.");
        true
    } else if (rst[4] / rst[0]) >= 0.5 {
        debug!("Error too large: {}/{}", rst[4], rst[0]);
        true
    } else {
        debug!("Continue.");
        false
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, sync::Arc};

    use super::*;
    use crate::instance::Instances;
    use attribute_info::NominalAttributeInfo;
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    #[test]
    fn test_basic() {
        tracing_subscriber::registry()
            .with(fmt::layer())
            .with(EnvFilter::new("ripper=debug"))
            .init();
        let mut attributes = vec![];
        attributes.push(Attribute {
            name: "a".to_string(),
            index: 0,
            attribute_info: Some(
                NominalAttributeInfo::new(
                    Some(vec![
                        "foo".to_string(),
                        "bar".to_string(),
                        "baz".to_string(),
                    ]),
                    "a",
                )
                .unwrap(),
            ),
        });
        attributes.push(Attribute {
            name: "b".to_string(),
            index: 1,
            attribute_info: Some(
                NominalAttributeInfo::new(
                    Some(vec!["x".to_string(), "y".to_string(), "z".to_string()]),
                    "b",
                )
                .unwrap(),
            ),
        });
        attributes.push(Attribute {
            name: "position".to_string(),
            index: 0,
            attribute_info: Some(
                NominalAttributeInfo::new(
                    Some(vec![
                        "first".to_string(),
                        "second".to_string(),
                        "third".to_string(),
                    ]),
                    "position",
                )
                .unwrap(),
            ),
        });
        let mut dataset = RefCell::new(Instances::new(RefCell::new(attributes)));
        let cloned_dataset = dataset.clone();
        dataset.borrow_mut().instances.push(RefCell::new(Instance {
            attribute_values: vec![0, 1, 1],
            dataset: Some(cloned_dataset),
        }));
        let cloned_dataset = dataset.clone();
        dataset.borrow_mut().instances.push(RefCell::new(Instance {
            attribute_values: vec![0, 1, 1],
            dataset: Some(cloned_dataset),
        }));
        let cloned_dataset = dataset.clone();
        dataset.borrow_mut().instances.push(RefCell::new(Instance {
            attribute_values: vec![1, 0, 0],
            dataset: Some(cloned_dataset),
        }));
        let cloned_dataset = dataset.clone();
        dataset.borrow_mut().instances.push(RefCell::new(Instance {
            attribute_values: vec![0, 0, 0],
            dataset: Some(cloned_dataset),
        }));
        let cloned_dataset = dataset.clone();
        dataset.borrow_mut().instances.push(RefCell::new(Instance {
            attribute_values: vec![0, 0, 0],
            dataset: Some(cloned_dataset),
        }));
        let mut classifier = Rip::new();
        classifier
            .build_classifier(&mut dataset.borrow_mut())
            .unwrap();
    }
}
