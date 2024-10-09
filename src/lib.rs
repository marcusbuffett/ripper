#![feature(type_alias_impl_trait)]
use rule::Rule;
use tracing::{debug, info, warn};

use rand::{self, rngs::StdRng};
use rand::{Rng, SeedableRng};
use std::rc::Rc;
use std::sync::OnceLock;
use std::{cell::RefCell, collections::HashMap};

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
    pub class: Option<String>,
    pub ruleset: RuleSet,
    pub distributions: Vec<Vec<f64>>,
    pub optimizations: usize,
    pub folds: usize,
    pub min_no: f64,
    pub total: f64,
    pub ruleset_stats: Vec<RuleStats>,
}

impl std::fmt::Debug for Rip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Rip {{ class: {:?} }}", self.class)
    }
}

const MAX_DL_SURPLUS: f64 = 64.0;

static RNG: OnceLock<StdRng> = OnceLock::new();

pub fn get_seeded_rng() -> StdRng {
    RNG.get_or_init(|| {
        println!("Seeding RNG");
        StdRng::seed_from_u64(42)
    })
    .clone()
}

impl Rip {
    pub fn new() -> Self {
        Rip {
            class: None,
            ruleset: RuleSet::new(),
            distributions: Vec::new(),
            optimizations: 10,
            folds: 3,
            min_no: 2.0,
            total: 0.0,
            ruleset_stats: Vec::new(),
        }
    }

    pub fn distribution_for_instance(&self, instance: &Instance) -> Vec<f64> {
        for (i, rule) in self.ruleset.rules.iter().enumerate() {
            if rule.covers(instance) {
                return self.distributions[i].clone();
            }
        }
        panic!("No rule found for this instance");
    }

    pub fn classify_instance(&self, instance: &Instance) -> usize {
        for rule in self.ruleset.rules.iter() {
            if rule.covers(instance) {
                return rule.consequent;
            }
        }
        panic!();
        // let dist = self.distribution_for_instance(instance);
        // dist.iter()
        //     .enumerate()
        //     .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        //     .unwrap()
        //     .0
    }

    pub fn build_classifier(&mut self, instances: &mut Instances) -> Result<(), String> {
        // Remove instances with missing class
        // don't need to do this, as they will all have a class index
        // instances.delete_with_missing_class();

        self.total = RuleStats::num_all_conditions(instances) as f64;
        debug!("Number of all possible conditions = {}", self.total);

        // Sort instances by class frequency
        let mut data = instances.clone();
        info!(
            "Before sorted by class frequency: {}",
            data.instances
                .iter()
                .enumerate()
                .map(|(i, inst)| (i * inst.borrow().class_value()))
                .sum::<usize>()
        );
        data.sort_by_class_freq_asc();
        info!(
            "Sorted by class frequency: {}",
            data.instances
                .iter()
                .enumerate()
                .map(|(i, inst)| (i * inst.borrow().class_value()))
                .sum::<usize>()
        );

        self.ruleset = RuleSet::new();
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
            let (rules, stats, remaining_data) =
                self.ruleset_for_one_class(&data, expected_fp_rate, class_index, default_dl);
            self.ruleset.rules.extend(rules);
            self.ruleset_stats.push(stats);
            data = remaining_data;
        }

        for rule in self.ruleset.rules.iter_mut() {
            rule.clean_up(&data.clone());
        }
        let mut default_rule = Rule::new();
        default_rule.consequent = data.num_classes() - 1;
        self.ruleset.rules.push(default_rule);
        debug!("Final classifier ruleset: {:?}", self.ruleset);

        warn!(
            "todo: distribution stuff?, {} rules, {} stats",
            self.ruleset.rules.len(),
            self.ruleset_stats.len()
        );
        for (z, one_class) in self.ruleset_stats.iter_mut().enumerate() {
            debug!(
                "Distributions for rule {}",
                one_class.distributions.as_ref().unwrap().len()
            );
            for (xyz, class_dist) in one_class
                .distributions
                .as_mut()
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                let mut class_dist = normalize(&class_dist);
                self.distributions.push(class_dist);
            }
        }
        debug!("Distributions len: {}", self.distributions.len());

        Ok(())
    }

    pub fn ruleset_for_one_class(
        &self,
        data: &Instances,
        expected_fp_rate: f64,
        class_index: usize,
        default_description_length: f64,
    ) -> (Vec<Rule>, RuleStats, Instances) {
        debug!("\n\n\nGenerating a ruleset to predict {}", class_index);
        // let mut stop = false;
        let mut ruleset: Vec<Rule> = vec![];

        let mut dl = default_description_length;
        let mut min_dl = default_description_length;

        let mut stats: Option<RuleStats> = None;
        let mut rst;
        let mut new_data: Instances = data.clone();
        let mut last_grow_data: Option<Instances> = None;
        let mut last_prune_data: Option<Instances> = None;

        let mut has_positive = true;

        loop {
            new_data = RuleStats::stratify(&new_data, self.folds);
            let (mut grow_data, prune_data) = RuleStats::partition(&new_data, self.folds);
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
                    let mut stats = RuleStats::new();
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

        for z in 0..self.optimizations {
            debug!("Run #{z} of optimizing ruleset {:?}", ruleset);

            let mut new_data = data.clone();
            let mut final_ruleset_stat = RuleStats::new();
            final_ruleset_stat.set_data(new_data.clone());
            final_ruleset_stat.set_num_all_conds(self.total);
            let mut position: i32 = 0;
            let mut stop = false;
            let mut is_residual;
            let mut has_positive = true;
            let mut dl = min_dl;
            let mut min_dl = default_description_length;

            while !stop && has_positive {
                debug!(?position, ?stop, ?position, ?ruleset);
                is_residual = position >= ruleset.len() as i32;
                new_data = RuleStats::stratify(&new_data, self.folds);
                let (mut grow_data, prune_data) = RuleStats::partition(&new_data, self.folds);

                debug!(
                    "\nRule #{} | isResidual? {} | data size: {}",
                    position,
                    is_residual,
                    new_data.instances.len()
                );

                let final_rule = if is_residual {
                    let mut new_rule = Rule::new();
                    new_rule.consequent = class_index;
                    debug!("Growing and pruning a new rule ...");
                    new_rule.grow(&mut grow_data).expect("Grow failed");
                    new_rule.prune(&prune_data, false);
                    debug!("New rule found: {:?}", new_rule);
                    new_rule
                } else {
                    let old_rule = &ruleset[position as usize];
                    let covers = new_data
                        .instances
                        .iter()
                        .any(|inst| old_rule.covers(&inst.borrow()));

                    if !covers {
                        final_ruleset_stat.add_and_update(old_rule);
                        position += 1;
                        continue;
                    }

                    debug!("Growing and pruning Replace ...");
                    let mut replace = Rule::new();
                    replace.consequent = class_index;
                    replace.grow(&mut grow_data).expect("Grow failed");
                    let prune_data = RuleStats::rm_covered_by_successives(
                        &prune_data,
                        &ruleset,
                        position as usize,
                    );
                    replace.prune(&prune_data, true);

                    debug!("Growing and pruning Revision ...");
                    let mut revision = old_rule.clone();
                    let mut new_grow_data = Instances::new(grow_data.attributes.clone());
                    new_grow_data.instances = grow_data
                        .instances
                        .iter()
                        .filter(|inst| revision.covers(&inst.borrow()))
                        .cloned()
                        .collect::<Vec<Rc<RefCell<Instance>>>>();
                    revision.grow(&mut new_grow_data).expect("Grow failed");
                    revision.prune(&prune_data, true);

                    let prev_rule_stats: Vec<&Vec<f64>> = (0..position)
                        .map(|c| final_ruleset_stat.get_simple_stats(c as usize))
                        .collect();

                    let mut temp_rules = ruleset.clone();
                    temp_rules[position as usize] = replace.clone();

                    let mut rep_stat =
                        RuleStats::new_with_data_and_rules(data.clone(), temp_rules.clone());
                    rep_stat.set_num_all_conds(self.total);
                    rep_stat.count_data_with(position as usize, &new_data, &prev_rule_stats);
                    let rst = rep_stat.get_simple_stats(position as usize);
                    debug!("Replace rule covers: {} | pos = {} | neg = {} \nThe rule doesn't cover: {} | pos = {}",
                           rst[0], rst[2], rst[4], rst[1], rst[5]);

                    let rep_dl = rep_stat.relative_dl(position as usize, expected_fp_rate);
                    debug!("Replace: {:?} |dl = {}", replace, rep_dl);

                    temp_rules[position as usize] = revision.clone();
                    let mut rev_stat = RuleStats::new_with_data_and_rules(data.clone(), temp_rules);
                    rev_stat.set_num_all_conds(self.total);
                    rev_stat.count_data_with(
                        position as usize,
                        &new_data,
                        &prev_rule_stats.as_slice(),
                    );
                    let rev_dl = rev_stat.relative_dl(position as usize, expected_fp_rate);
                    debug!("Revision: {:?} |dl = {}", revision, rev_dl);

                    let mut rstats =
                        RuleStats::new_with_data_and_rules(data.clone(), ruleset.clone());
                    rstats.set_num_all_conds(self.total);
                    rstats.count_data_with(
                        position as usize,
                        &new_data,
                        prev_rule_stats.as_slice(),
                    );
                    let old_dl = rstats.relative_dl(position as usize, expected_fp_rate);
                    debug!("Old rule: {:?} |dl = {}", old_rule, old_dl);

                    debug!(
                        "\nrepDL: {} \nrevDL: {} \noldDL: {}",
                        rep_dl, rev_dl, old_dl
                    );
                    debug!(
                        "Old rule: {:#?}, replace rule: {:#?}, revision rule: {:#?}",
                        old_rule, replace, revision
                    );

                    if old_dl <= rev_dl && old_dl <= rep_dl {
                        old_rule.clone()
                    } else if rev_dl <= rep_dl {
                        revision
                    } else {
                        replace
                    }
                };

                final_ruleset_stat.add_and_update(&final_rule);
                let rst = final_ruleset_stat
                    .get_simple_stats(position as usize)
                    .clone();

                if is_residual {
                    dl += final_ruleset_stat.relative_dl(position as usize, expected_fp_rate);
                    debug!("After optimization: the dl = {} | best: {}", dl, min_dl);

                    if dl < min_dl {
                        min_dl = dl;
                    }

                    stop = check_stop(&rst, min_dl, dl);
                    if !stop {
                        ruleset.push(final_rule);
                    } else {
                        final_ruleset_stat.remove_last();
                        position -= 1;
                    }
                } else {
                    ruleset[position as usize] = final_rule;
                }

                debug!("The rule covers: {} | pos = {} | neg = {} \nThe rule doesn't cover: {} | pos = {}",
                       rst[0], rst[2], rst[4], rst[1], rst[5]);
                debug!("Ruleset so far: {:?}", ruleset);

                if final_ruleset_stat.get_ruleset_size() > 0 {
                    new_data = final_ruleset_stat.get_filtered(position as usize).1.clone();
                }
                has_positive = rst[5] > 0.0;
                position += 1;
            }

            if ruleset.len() > ((position + 1) as usize) {
                for k in ((position + 1) as usize)..ruleset.len() {
                    final_ruleset_stat.add_and_update(&ruleset[k]);
                }
            }

            debug!("Deleting rules to decrease DL of the whole ruleset ...");
            final_ruleset_stat.reduce_dl(expected_fp_rate);
            let del = ruleset.len() - final_ruleset_stat.get_ruleset_size();
            debug!("{} rules are deleted after DL reduction procedure", del);

            ruleset = final_ruleset_stat
                .get_ruleset()
                .expect("Should have ruleset")
                .to_vec();
            stats = Some(final_ruleset_stat);
        }

        debug!("Final ruleset for {}: {:?}", class_index, ruleset);

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

    pub fn evaluate(&self, borrow: &Instances) -> f64 {
        let mut num_correct = 0;
        let num_total = borrow.instances.len();
        for inst in borrow.instances.iter() {
            let predicted = self.classify_instance(&inst.borrow());
            if inst.borrow().class_value() == predicted {
                num_correct += 1;
            }
        }
        let accuracy = (num_correct as f64 / num_total as f64);
        println!("Accuracy: {}%", accuracy * 100.0);
        return accuracy;
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
    use std::rc::Rc;
    use std::{cell::RefCell, sync::Arc};

    use super::*;
    use crate::instance::Instances;
    use attribute_info::NominalAttributeInfo;
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    use std::collections::{BTreeSet, HashMap, HashSet};
    use std::fs::File;
    use std::path::Path;

    fn csv_to_instances<P: AsRef<Path> + Clone>(file_path: P) -> Rc<RefCell<Instances>> {
        let file = File::open(file_path.clone()).expect("Failed to open CSV file");
        let mut reader = csv::Reader::from_reader(file);

        let headers = reader
            .headers()
            .expect("Failed to read CSV headers")
            .clone();

        // First pass: collect unique values for each column
        let mut column_values: Vec<BTreeSet<String>> = vec![BTreeSet::new(); headers.len()];
        for result in reader.records() {
            let record = result.expect("Failed to read CSV record");
            for (i, field) in record.iter().enumerate() {
                column_values[i].insert(field.to_string());
            }
        }

        // Create attributes
        let mut attributes = Vec::new();
        for (i, header) in headers.iter().enumerate() {
            let values: Vec<String> = column_values[i].iter().cloned().collect();
            let attribute_info = NominalAttributeInfo::new(Some(values), header)
                .expect("Failed to create NominalAttributeInfo");

            attributes.push(Attribute {
                name: header.to_string(),
                index: i,
                attribute_info: Some(attribute_info),
            });
        }

        let mut dataset = Rc::new(RefCell::new(Instances::new(Rc::new(RefCell::new(
            attributes,
        )))));

        // Second pass: add instances
        let mut reader =
            csv::Reader::from_reader(File::open(file_path).expect("Failed to open CSV file"));
        debug!("Created reader!");
        for result in reader.records() {
            let record = result.expect("Failed to read CSV record");
            let mut attribute_values = Vec::new();

            for (i, field) in record.iter().enumerate() {
                let attributes = &dataset.borrow().attributes;
                let attr = &attributes.borrow()[i];
                let value_index = attr
                    .attribute_info
                    .as_ref()
                    .expect("Attribute info is missing")
                    .hashtable
                    .get(field)
                    .expect("Failed to find value in attribute hashtable")
                    .to_owned();
                attribute_values.push(value_index);
            }

            let cloned_dataset = dataset.clone();
            dataset
                .borrow_mut()
                .instances
                .push(Rc::new(RefCell::new(Instance {
                    attribute_values,
                    dataset: Some(cloned_dataset),
                })));
        }

        debug!("Created instances!");
        dataset
    }

    #[test]
    fn test_csv() {
        tracing_subscriber::registry()
            .with(fmt::layer())
            .with(EnvFilter::new("ripper=info"))
            .init();
        let dataset = csv_to_instances("./dragon.csv");
        let mut accuracies = vec![];
        for i in 0..10 {
            let mut classifier = Rip::new();
            classifier
                .build_classifier(&mut dataset.borrow_mut())
                .unwrap();
            accuracies.push(classifier.evaluate(&dataset.borrow()));
        }
        println!(
            "Average accuracy: {}",
            accuracies.iter().sum::<f64>() / accuracies.len() as f64 * 100.
        );
        // dbg!(&dataset);
    }

    // #[test]
    // fn test_basic() {
    //     tracing_subscriber::registry()
    //         .with(fmt::layer())
    //         .with(EnvFilter::new("ripper=debug"))
    //         .init();
    //     let mut attributes = vec![];
    //     attributes.push(Attribute {
    //         name: "a".to_string(),
    //         index: 0,
    //         attribute_info: Some(
    //             NominalAttributeInfo::new(
    //                 Some(vec![
    //                     "foo".to_string(),
    //                     "bar".to_string(),
    //                     "baz".to_string(),
    //                 ]),
    //                 "a",
    //             )
    //             .unwrap(),
    //         ),
    //     });
    //     attributes.push(Attribute {
    //         name: "b".to_string(),
    //         index: 1,
    //         attribute_info: Some(
    //             NominalAttributeInfo::new(
    //                 Some(vec!["x".to_string(), "y".to_string(), "z".to_string()]),
    //                 "b",
    //             )
    //             .unwrap(),
    //         ),
    //     });
    //     attributes.push(Attribute {
    //         name: "position".to_string(),
    //         index: 0,
    //         attribute_info: Some(
    //             NominalAttributeInfo::new(
    //                 Some(vec!["first".to_string(), "second".to_string()]),
    //                 "position",
    //             )
    //             .unwrap(),
    //         ),
    //     });
    //     let mut dataset = RefCell::new(Instances::new(RefCell::new(attributes)));
    //     let cloned_dataset = dataset.clone();
    //     dataset.borrow_mut().instances.push(RefCell::new(Instance {
    //         attribute_values: vec![0, 1, 0],
    //         dataset: Some(cloned_dataset),
    //     }));
    //     let cloned_dataset = dataset.clone();
    //     dataset.borrow_mut().instances.push(RefCell::new(Instance {
    //         attribute_values: vec![0, 1, 0],
    //         dataset: Some(cloned_dataset),
    //     }));
    //     let cloned_dataset = dataset.clone();
    //     dataset.borrow_mut().instances.push(RefCell::new(Instance {
    //         attribute_values: vec![1, 1, 1],
    //         dataset: Some(cloned_dataset),
    //     }));
    //     let cloned_dataset = dataset.clone();
    //     dataset.borrow_mut().instances.push(RefCell::new(Instance {
    //         attribute_values: vec![1, 1, 1],
    //         dataset: Some(cloned_dataset),
    //     }));
    //     let cloned_dataset = dataset.clone();
    //     dataset.borrow_mut().instances.push(RefCell::new(Instance {
    //         attribute_values: vec![1, 1, 1],
    //         dataset: Some(cloned_dataset),
    //     }));
    //     let cloned_dataset = dataset.clone();
    //     dataset.borrow_mut().instances.push(RefCell::new(Instance {
    //         attribute_values: vec![1, 1, 1],
    //         dataset: Some(cloned_dataset),
    //     }));
    //     let cloned_dataset = dataset.clone();
    //     dataset.borrow_mut().instances.push(RefCell::new(Instance {
    //         attribute_values: vec![0, 1, 1],
    //         dataset: Some(cloned_dataset),
    //     }));
    //     let cloned_dataset = dataset.clone();
    //     dataset.borrow_mut().instances.push(RefCell::new(Instance {
    //         attribute_values: vec![1, 1, 1],
    //         dataset: Some(cloned_dataset),
    //     }));
    //     let cloned_dataset = dataset.clone();
    //     dataset.borrow_mut().instances.push(RefCell::new(Instance {
    //         attribute_values: vec![0, 0, 0],
    //         dataset: Some(cloned_dataset),
    //     }));
    //     let cloned_dataset = dataset.clone();
    //     dataset.borrow_mut().instances.push(RefCell::new(Instance {
    //         attribute_values: vec![0, 0, 0],
    //         dataset: Some(cloned_dataset),
    //     }));
    //     let mut classifier = Rip::new();
    //     classifier
    //         .build_classifier(&mut dataset.borrow_mut())
    //         .unwrap();
    // }
}

fn normalize(vec: &Vec<f64>) -> Vec<f64> {
    let magnitude: f64 = vec.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if magnitude == 0.0 {
        vec.clone()
    } else {
        vec.iter().map(|&x| x / magnitude).collect()
    }
}
