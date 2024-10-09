// rule_stats.rs

use rand::seq::SliceRandom;
use tracing::debug;

use crate::{get_seeded_rng, Instances, Rule};

const REDUNDANCY_FACTOR: f64 = 0.5;
const MDL_THEORY_WEIGHT: f64 = 1.0;

#[derive(Debug, Clone)]
pub struct RuleStats {
    pub data: Option<Instances>,
    pub ruleset: Option<Vec<Rule>>,
    pub simple_stats: Option<Vec<Vec<f64>>>,
    pub filtered_instances: Option<Vec<(Instances, Instances)>>,
    pub total: f64,
    pub mdl_theory_weight: f64,
    pub distributions: Option<Vec<Vec<f64>>>,
}

impl RuleStats {
    pub fn new() -> Self {
        RuleStats {
            data: None,
            ruleset: None,
            simple_stats: None,
            filtered_instances: None,
            total: -1.0,
            mdl_theory_weight: 1.0,
            distributions: None,
        }
    }

    pub fn new_with_data_and_rules(data: Instances, rules: Vec<Rule>) -> Self {
        let mut stats = Self::new();
        stats.data = Some(data);
        stats.ruleset = Some(rules);
        stats
    }

    pub fn clean_up(&mut self) {
        self.data = None;
        self.filtered_instances = None;
    }

    pub fn set_num_all_conds(&mut self, total: f64) {
        if total < 0.0 {
            self.total = Self::num_all_conditions(&self.data.as_ref().unwrap()) as f64;
        } else {
            self.total = total;
        }
    }

    pub fn set_data(&mut self, data: Instances) {
        self.data = Some(data);
    }

    pub fn get_data(&self) -> Option<&Instances> {
        self.data.as_ref()
    }

    pub fn set_ruleset(&mut self, rules: Vec<Rule>) {
        self.ruleset = Some(rules);
    }

    pub fn get_ruleset(&self) -> Option<&Vec<Rule>> {
        self.ruleset.as_ref()
    }

    pub fn get_ruleset_size(&self) -> usize {
        self.ruleset.as_ref().map_or(0, |r| r.len())
    }

    pub fn get_simple_stats(&self, index: usize) -> &Vec<f64> {
        self.simple_stats
            .as_ref()
            .and_then(|stats| stats.get(index))
            .expect("No simple stats for this index")
    }

    pub fn get_filtered(&self, index: usize) -> &(Instances, Instances) {
        self.filtered_instances
            .as_ref()
            .expect("No filtered instances")
            .get(index)
            .expect("No filtered instances for this index")
    }

    pub fn get_distributions(&self, index: usize) -> Option<&Vec<f64>> {
        self.distributions.as_ref().and_then(|dist| dist.get(index))
    }

    pub fn set_mdl_theory_weight(&mut self, weight: f64) {
        self.mdl_theory_weight = weight;
    }

    pub fn num_all_conditions(data: &Instances) -> usize {
        let attributes = data.attributes.borrow();
        let total = attributes.iter().map(|a| a.num_values()).sum::<usize>();
        total
    }

    pub fn count_data(&mut self) {
        if self.filtered_instances.is_some() || self.ruleset.is_none() || self.data.is_none() {
            return;
        }

        let size = self.ruleset.as_ref().unwrap().len();
        self.filtered_instances = Some(Vec::with_capacity(size));
        self.simple_stats = Some(Vec::with_capacity(size));
        self.distributions = Some(Vec::with_capacity(size));
        let mut data = self.data.as_ref().unwrap().clone();

        for i in 0..(size) {
            let mut stats = [0.0; 6]; // 6 statistics parameters
            let mut class_counts = vec![0.0; self.data.as_ref().unwrap().num_classes()];
            let filtered = self.compute_simple_stats(i, &data, &mut stats, Some(&mut class_counts));
            self.filtered_instances
                .as_mut()
                .unwrap()
                .push(filtered.clone());
            self.simple_stats.as_mut().unwrap().push(stats.to_vec());
            self.distributions.as_mut().unwrap().push(class_counts);
            data = filtered.1.clone(); // Data not covered
        }
    }

    pub fn count_data_from(
        &mut self,
        index: usize,
        uncovered: &Instances,
        prev_rule_stats: &[Vec<f64>],
    ) {
        // Implementation of countData with index
        unimplemented!()
    }

    pub fn add_and_update(&mut self, last_rule: &Rule) {
        if self.ruleset.is_none() {
            self.ruleset = Some(vec![]);
        }
        let ruleset = self.ruleset.as_mut().unwrap();
        ruleset.push(last_rule.clone());
        let data: Instances = if self.filtered_instances.is_none() {
            self.data.as_ref().unwrap().clone()
        } else {
            let filtered_instances = self.filtered_instances.as_ref().unwrap();
            filtered_instances
                .last()
                .expect("No filtered instances")
                .1
                .clone()
        };
        let mut stats = vec![0.0; 6];
        let mut class_counts = vec![0.0; data.num_classes()];
        let ruleset_len = ruleset.len();
        drop(ruleset);
        let filtered =
            self.compute_simple_stats(ruleset_len - 1, &data, &mut stats, Some(&mut class_counts));
        if self.filtered_instances.is_none() {
            self.filtered_instances = Some(vec![]);
        }
        self.filtered_instances.as_mut().unwrap().push(filtered);
        if self.simple_stats.is_none() {
            self.simple_stats = Some(vec![]);
        }
        self.simple_stats.as_mut().unwrap().push(stats);
        if self.distributions.is_none() {
            self.distributions = Some(vec![]);
        }
        self.distributions.as_mut().unwrap().push(class_counts);
    }

    pub fn theory_dl(&self, index: usize) -> f64 {
        let k = self.ruleset.as_ref().unwrap()[index].antds.len() as f64;

        if k == 0.0 {
            return 0.0;
        }

        let mut tdl = k.log2();
        if k > 1.0 {
            tdl += 2.0 * tdl.log2(); // of log2 star
        }
        tdl += Self::subset_dl(self.total, k, k / self.total);

        // Uncomment for debugging
        // println!("!!!theory: {}", MDL_THEORY_WEIGHT * REDUNDANCY_FACTOR * tdl);

        MDL_THEORY_WEIGHT * REDUNDANCY_FACTOR * tdl
    }

    pub fn data_dl(exp_fp_over_err: f64, cover: f64, uncover: f64, fp: f64, fn_: f64) -> f64 {
        let total_bits = (cover + uncover + 1.0).log2(); // how many data?
        let (cover_bits, uncover_bits); // What's the error?
        let exp_err; // Expected FP or FN

        if cover > uncover {
            exp_err = exp_fp_over_err * (fp + fn_);
            cover_bits = Self::subset_dl(cover, fp, exp_err / cover);
            uncover_bits = if uncover > 0.0 {
                Self::subset_dl(uncover, fn_, fn_ / uncover)
            } else {
                0.0
            };
        } else {
            exp_err = (1.0 - exp_fp_over_err) * (fp + fn_);
            cover_bits = if cover > 0.0 {
                Self::subset_dl(cover, fp, fp / cover)
            } else {
                0.0
            };
            uncover_bits = Self::subset_dl(uncover, fn_, exp_err / uncover);
        }

        // Uncomment for debugging
        /*
        println!("!!!cover: {}|uncover{}|coverBits: {}|uncBits: {}|\
                  FPRate: {}|expErr: {}|fp: {}|fn: {}|total: {}",
                 cover, uncover, cover_bits, uncover_bits,
                 exp_fp_over_err, exp_err, fp, fn_, total_bits);
        */

        total_bits + cover_bits + uncover_bits
    }

    pub fn potential(
        &self,
        index: usize,
        exp_fp_over_err: f64,
        ruleset_stat: &mut [f64],
        rule_stat: &[f64],
    ) -> f64 {
        // Restore the stats if deleted
        let pcov = ruleset_stat[0] - rule_stat[0];
        let puncov = ruleset_stat[1] + rule_stat[0];
        let pfp = ruleset_stat[4] - rule_stat[4];
        let pfn = ruleset_stat[5] + rule_stat[2];

        let data_dl_with = Self::data_dl(
            exp_fp_over_err,
            ruleset_stat[0],
            ruleset_stat[1],
            ruleset_stat[4],
            ruleset_stat[5],
        );
        let theory_dl_with = self.theory_dl(index);
        let data_dl_without = Self::data_dl(exp_fp_over_err, pcov, puncov, pfp, pfn);

        let potential = data_dl_with + theory_dl_with - data_dl_without;
        let err = rule_stat[4] / rule_stat[0];

        // Uncomment for debugging
        /*
        println!("!!!{} | {} | {}|{} / {}",
                 data_dl_with, theory_dl_with, data_dl_without, rule_stat[4], rule_stat[0]);
        */

        let over_err = err >= 0.5;

        if potential >= 0.0 || over_err {
            // If deleted, update ruleset stats. Other stats do not matter
            ruleset_stat[0] = pcov;
            ruleset_stat[1] = puncov;
            ruleset_stat[4] = pfp;
            ruleset_stat[5] = pfn;
            potential
        } else {
            f64::NAN
        }
    }

    pub fn min_data_dl_if_deleted(&self, index: usize, exp_fp_rate: f64) -> f64 {
        let mut ruleset_stat = [0.0; 6]; // Stats of ruleset if deleted
        let more = self.ruleset.as_ref().expect("Should be a ruleset").len() - 1 - index; // How many rules after?
        let mut index_plus = Vec::with_capacity(more); // Their stats
        let simple_stats = self.simple_stats.as_ref().unwrap();

        // 0...(index-1) are OK
        for j in 0..index {
            // Covered stats are cumulative
            ruleset_stat[0] += simple_stats[j][0];
            ruleset_stat[2] += simple_stats[j][2];
            ruleset_stat[4] += simple_stats[j][4];
        }

        // Recount data from index+1
        let mut data = if index == 0 {
            self.data.as_ref().unwrap().clone()
        } else {
            self.filtered_instances.as_ref().unwrap()[index - 1]
                .1
                .clone()
        };
        // println!("!!!without: {}", data.sum_of_weights());

        for j in (index + 1)..self.ruleset.as_ref().expect("Should be a ruleset").len() {
            let mut stats = [0.0; 6];
            let split = self.compute_simple_stats(j, &data, &mut stats, None);
            index_plus.push(stats);
            ruleset_stat[0] += stats[0];
            ruleset_stat[2] += stats[2];
            ruleset_stat[4] += stats[4];
            data = split.1.clone();
        }

        // Uncovered stats are those of the last rule
        if more > 0 {
            let last = index_plus.last().unwrap();
            ruleset_stat[1] = last[1];
            ruleset_stat[3] = last[3];
            ruleset_stat[5] = last[5];
        } else if index > 0 {
            ruleset_stat[1] = simple_stats[index - 1][1];
            ruleset_stat[3] = simple_stats[index - 1][3];
            ruleset_stat[5] = simple_stats[index - 1][5];
        } else {
            // Null coverage
            ruleset_stat[1] = simple_stats[0][0] + simple_stats[0][1];
            ruleset_stat[3] = simple_stats[0][3] + simple_stats[0][4];
            ruleset_stat[5] = simple_stats[0][2] + simple_stats[0][5];
        }

        // Potential
        let mut potential = 0.0;
        for k in (index + 1)..self.ruleset.as_ref().expect("Should be a ruleset").len() {
            let rule_stat = &index_plus[k - index - 1];
            let if_deleted = self.potential(k, exp_fp_rate, &mut ruleset_stat, rule_stat);
            if !if_deleted.is_nan() {
                potential += if_deleted;
            }
        }

        // Data DL of the ruleset without the rule
        // Note that ruleset stats has already been updated to reflect
        // deletion if any potential
        let data_dl_without = Self::data_dl(
            exp_fp_rate,
            ruleset_stat[0],
            ruleset_stat[1],
            ruleset_stat[4],
            ruleset_stat[5],
        );
        // println!("!!!without: {} |potential: {}", data_dl_without, potential);

        // Why subtract potential again? To reflect change of theory DL??
        data_dl_without - potential
    }

    pub fn min_data_dl_if_exists(&self, index: usize, exp_fp_rate: f64) -> f64 {
        let mut ruleset_stat = [0.0; 6]; // Stats of ruleset if rule exists
        let simple_stats = self.simple_stats.as_ref().unwrap();
        for j in 0..simple_stats.len() {
            // Covered stats are cumulative
            ruleset_stat[0] += simple_stats[j][0];
            ruleset_stat[2] += simple_stats[j][2];
            ruleset_stat[4] += simple_stats[j][4];
            if j == simple_stats.len() - 1 {
                // Last rule
                ruleset_stat[1] = simple_stats[j][1];
                ruleset_stat[3] = simple_stats[j][3];
                ruleset_stat[5] = simple_stats[j][5];
            }
        }

        // Potential
        let mut potential = 0.0;
        for k in index + 1..simple_stats.len() {
            // todo: performance improvement possible
            let rule_stat = self.get_simple_stats(k).clone();
            let if_deleted = self.potential(k, exp_fp_rate, &mut ruleset_stat, &rule_stat);
            if !if_deleted.is_nan() {
                potential += if_deleted;
            }
        }

        // Data DL of the ruleset without the rule
        // Note that ruleset stats has already been updated to reflect deletion
        // if any potential
        let data_dl_with = Self::data_dl(
            exp_fp_rate,
            ruleset_stat[0],
            ruleset_stat[1],
            ruleset_stat[4],
            ruleset_stat[5],
        );

        data_dl_with - potential
    }

    pub fn relative_dl(&self, index: usize, exp_fp_rate: f64) -> f64 {
        return self.min_data_dl_if_exists(index, exp_fp_rate) + self.theory_dl(index)
            - self.min_data_dl_if_deleted(index, exp_fp_rate);
    }

    pub fn reduce_dl(&mut self, exp_fp_rate: f64) {
        let mut need_update = false;
        let mut ruleset_stat = [0.0; 6];
        let simple_stats = self
            .simple_stats
            .as_ref()
            .expect("Should have stats")
            .clone();

        for (j, simple_stat) in simple_stats.iter().enumerate() {
            // Covered stats are cumulative
            ruleset_stat[0] += simple_stat[0];
            ruleset_stat[2] += simple_stat[2];
            ruleset_stat[4] += simple_stat[4];
            if j == simple_stats.len() - 1 {
                // Last rule
                ruleset_stat[1] = simple_stat[1];
                ruleset_stat[3] = simple_stat[3];
                ruleset_stat[5] = simple_stat[5];
            }
        }

        // Potential
        let mut to_remove = Vec::new();
        for k in (0..simple_stats.len()).rev() {
            let rule_stat = simple_stats.get(k).expect("Should have simple stats at k");

            // ruleset_stat updated
            let if_deleted = self.potential(k, exp_fp_rate, &mut ruleset_stat, &rule_stat);
            if !if_deleted.is_nan() {
                // Uncomment for debugging
                // println!("!!!deleted ({}): save {} | {} | {} | {} | {}",
                //          k, if_deleted, ruleset_stat[0], ruleset_stat[1], ruleset_stat[4], ruleset_stat[5]);

                if k == simple_stats.len() - 1 {
                    self.remove_last();
                } else {
                    to_remove.push(k);
                    need_update = true;
                }
            }
        }

        // Remove rules in reverse order to maintain correct indices
        for &k in to_remove.iter() {
            self.ruleset
                .as_mut()
                .expect("Should have ruleset")
                .remove(k);
        }

        if need_update {
            self.filtered_instances = None;
            self.simple_stats = None;
            self.count_data();
        }
    }

    pub fn remove_last(&mut self) {
        if let Some(ruleset) = self.ruleset.as_mut() {
            ruleset.pop();
        }
        if let Some(filtered) = self.filtered_instances.as_mut() {
            filtered.pop();
        }
        if let Some(simple_stats) = self.simple_stats.as_mut() {
            simple_stats.pop();
        }
        if let Some(distributions) = self.distributions.as_mut() {
            distributions.pop();
        }
    }

    pub fn rm_covered_by_successives(data: &Instances, rules: &[Rule], index: usize) -> Instances {
        let mut rt = Instances::new(data.attributes.clone());

        for datum in &data.instances {
            let covered = rules
                .iter()
                .skip(index + 1)
                .any(|rule| rule.covers(&datum.borrow()));

            if !covered {
                rt.instances.push(datum.clone());
            }
        }

        rt
    }

    pub fn stratify(data: &Instances, folds: usize) -> Instances {
        let mut result: Instances = Instances::new(data.attributes.clone());
        let mut bags_by_classes: Vec<Instances> = (0..data.num_classes())
            .map(|_| Instances::new(data.attributes.clone()))
            .collect();

        for datum in data.instances.iter() {
            bags_by_classes[datum.borrow().class_value()]
                .instances
                .push(datum.clone());
        }
        // TODO: this may be sorting by an unsorted class order

        // shuffle each bag
        for bag in bags_by_classes.iter_mut() {
            bag.instances.shuffle(&mut get_seeded_rng());
        }

        // fold stuff
        for k in 0..folds {
            let mut offset = k;
            let mut bag = 0;
            'one_fold: loop {
                while offset >= bags_by_classes[bag].instances.len() {
                    offset -= bags_by_classes[bag].instances.len();
                    bag += 1;
                    if bag >= bags_by_classes.len() {
                        break 'one_fold;
                    }
                }

                result
                    .instances
                    .push(bags_by_classes[bag].instances[offset].clone());
                offset += folds;
            }
        }

        result
    }

    pub fn combined_dl(&self, exp_fp_rate: f64, predicted: f64) -> f64 {
        // Implementation of combinedDL
        unimplemented!()
    }

    pub fn partition(data: &Instances, num_folds: usize) -> (Instances, Instances) {
        let splits = data.instances.len() * (num_folds - 1) / num_folds;

        let first_partition = Instances {
            attributes: data.attributes.clone(),
            instances: data.instances[0..splits].to_vec(),
        };

        let second_partition = Instances {
            attributes: data.attributes.clone(),
            instances: data.instances[splits..].to_vec(),
        };

        (first_partition, second_partition)
    }

    pub fn subset_dl(t: f64, k: f64, p: f64) -> f64 {
        let mut rt = if p > 0.0 { -k * p.log2() } else { 0.0 };
        rt -= (t - k) * (1.0 - p).log2();
        rt
    }

    pub fn data_description_length(
        exp_fp_over_err: f64,
        cover: f64,
        uncover: f64,
        fp: f64,
        fn_val: f64,
    ) -> f64 {
        let total_bits = (cover + uncover + 1.0).log2();
        let (cover_bits, uncover_bits, exp_err);

        if cover > uncover {
            exp_err = exp_fp_over_err * (fp + fn_val);
            cover_bits = Self::subset_dl(cover, fp, exp_err / cover);
            uncover_bits = if uncover > 0.0 {
                Self::subset_dl(uncover, fn_val, fn_val / uncover)
            } else {
                0.0
            };
        } else {
            exp_err = (1.0 - exp_fp_over_err) * (fp + fn_val);
            cover_bits = if cover > 0.0 {
                Self::subset_dl(cover, fp, fp / cover)
            } else {
                0.0
            };
            uncover_bits = Self::subset_dl(uncover, fn_val, exp_err / uncover);
        }

        // Uncomment for debugging:
        // eprintln!(
        //     "!!!cover: {}|uncover: {}|coverBits: {}|uncBits: {}|FPRate: {}|expErr: {}|fp: {}|fn: {}|total: {}",
        //     cover, uncover, cover_bits, uncover_bits, exp_fp_over_err, exp_err, fp, fn_val, total_bits
        // );

        total_bits + cover_bits + uncover_bits
    }

    pub fn compute_simple_stats(
        self: &Self,
        idx: usize,
        insts: &Instances,
        stats: &mut [f64],
        mut dist: Option<&mut [f64]>,
    ) -> (Instances, Instances) {
        let ruleset = &self.ruleset.as_ref().expect("RuleSet not set");
        let rule = ruleset.get(idx).expect("Rule not found");

        let mut covered_instances = Instances::new(insts.attributes.clone());
        let mut not_covered_instances = Instances::new(insts.attributes.clone());

        for datum in insts.instances.iter() {
            // let weight = datum.weight();
            if rule.covers(&datum.borrow()) {
                covered_instances.instances.push(datum.clone());
                stats[0] += 1.0; // Coverage
                if datum.borrow().class_value() == rule.consequent {
                    stats[2] += 1.0; // True positives
                } else {
                    stats[4] += 1.0; // False positives
                }
                if let Some(dist) = dist.as_deref_mut() {
                    (dist)[datum.borrow().class_value() as usize] += 1.;
                }
            } else {
                not_covered_instances.instances.push(datum.clone());
                stats[1] += 1.0;
                if datum.borrow().class_value() != rule.consequent {
                    stats[3] += 1.0; // True negatives
                } else {
                    stats[5] += 1.0; // False negatives
                }
            }
        }

        (covered_instances, not_covered_instances)
    }

    pub fn count_data_with(
        &mut self,
        index: usize,
        uncovered: &Instances,
        prev_rule_stats: &[&Vec<f64>],
    ) {
        if self.filtered_instances.is_some() || self.ruleset.is_none() {
            debug!(
                "Returning early, filtered instances is {} and ruleset is {}",
                if self.filtered_instances.is_some() {
                    "not None"
                } else {
                    "None"
                },
                if self.ruleset.is_some() {
                    "not None"
                } else {
                    "None"
                }
            );
            return;
        }

        let size = self.ruleset.as_ref().unwrap().len();
        self.filtered_instances = Some(Vec::with_capacity(size));
        self.simple_stats = Some(Vec::with_capacity(size));
        let mut data: (Instances, Instances) = (
            Instances::new(uncovered.attributes.clone()),
            uncovered.clone(),
        );

        for i in 0..index {
            self.simple_stats
                .as_mut()
                .unwrap()
                .push(prev_rule_stats[i].to_vec());
            if i + 1 == index {
                self.filtered_instances.as_mut().unwrap().push((
                    Instances::new(uncovered.attributes.clone()),
                    uncovered.clone(),
                ));
            } else {
                // idk they're pushing empty stuff
                self.filtered_instances.as_mut().unwrap().push((
                    Instances::new(uncovered.attributes.clone()),
                    Instances::new(uncovered.attributes.clone()),
                ));
            }
        }

        for j in index..size {
            let mut stats = [0.0; 6]; // 6 statistics parameters
            let filtered = self.compute_simple_stats(j, &data.1, &mut stats, None);
            self.filtered_instances
                .as_mut()
                .unwrap()
                .push(filtered.clone());
            self.simple_stats.as_mut().unwrap().push(stats.to_vec());
            data = filtered; // Data not covered
        }
    }
}
