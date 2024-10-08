use itertools::Itertools;
use tracing::{debug, Level};

use crate::{
    antd::NominalAntd,
    instance::{Instance, Instances},
};

#[derive(Clone)]
pub struct Rule {
    pub consequent: usize,
    pub antds: Vec<NominalAntd>,
    pub min_no: usize,
}

impl std::fmt::Debug for Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}) => {}",
            self.antds
                .iter()
                .map(|antd| {
                    let attr_info = antd
                        .attr
                        .attribute_info
                        .as_ref()
                        .expect("Should have attribute info");
                    format!(
                        "{} = {}",
                        antd.attr.name,
                        attr_info.values.get(antd.value).unwrap()
                    )
                })
                .join(" & "),
            self.consequent,
        )
    }
}

impl Rule {
    pub fn new() -> Self {
        Rule {
            consequent: 0,
            antds: Vec::new(),
            min_no: 2,
        }
    }

    pub fn grow(&mut self, data: &mut Instances) -> Result<(), String> {
        let mut grow_data = data.clone();
        let sum_of_weights = grow_data.instances.len();
        if !(sum_of_weights > 0) {
            debug!("Not enough instances to grow");
            return Ok(());
        }

        // Compute the default accurate rate of the growing data
        let def_accu = self.compute_def_accu(&grow_data);
        let mut def_ac_rt = (def_accu + 1.0) / (sum_of_weights as f64 + 1.0);

        // Keep the record of which attributes have already been used
        let mut used = vec![false; grow_data.attributes.borrow().len()];
        let mut num_unused = used.len();

        // If there are already antecedents existing
        for antd in &self.antds {
            used[antd.attr.index] = true;
            num_unused -= 1;
        }

        while grow_data.instances.len() > 0 && (num_unused > 0) && (def_ac_rt < 1.0) {
            let mut max_info_gain = 0.0;
            let mut one_antd: Option<NominalAntd> = None;
            let mut cover_data: Option<Instances> = None;

            // Build one condition based on all attributes not used yet
            for (i, attr) in grow_data.attributes.borrow().iter().enumerate() {
                debug!("One condition: size = {}", grow_data.instances.len());

                // todo: performance opportunity
                let mut antd: NominalAntd = NominalAntd::new(attr.clone());
                if !used[i] {
                    let covered_data = self.compute_info_gain(&grow_data, def_ac_rt, &mut antd);

                    let info_gain = antd.max_info_gain;
                    debug!(
                        "Test of '{:?}': infoGain = {} | Accuracy = {}={}/{} def. accuracy: {}",
                        antd, info_gain, antd.accu_rate, antd.accu, antd.cover, def_ac_rt
                    );

                    if info_gain > max_info_gain {
                        one_antd = Some(antd);
                        cover_data = Some(covered_data);
                        max_info_gain = info_gain;
                    }
                }
            }

            let one_antd = match one_antd {
                Some(antd) => antd,
                None => break, // Cannot find antds
            };

            if (one_antd.accu as usize) < self.min_no {
                break; // Too low coverage
            }

            // Numeric attributes can be used more than once
            used[one_antd.attr.index] = true;
            num_unused -= 1;

            self.antds.push(one_antd.clone());
            grow_data = cover_data.unwrap(); // Grow data size is shrinking
            def_ac_rt = one_antd.accu_rate;
        }

        Ok(())
    }

    pub fn prune(&mut self, prune_data: &Instances, use_whole: bool) {
        let mut data = prune_data.clone();

        let total = data.instances.len();
        if !(total > 0) {
            return;
        }

        // The default accurate # and rate on pruning data
        let def_accu = self.compute_def_accu(&data);

        debug!(
            "Pruning with default accuracy {}, across {} negative instances: \n{:#?}",
            def_accu, total, data.instances
        );

        let size = self.antds.len();
        if size == 0 {
            return; // Default rule before pruning
        }

        let mut worth_rt = vec![0.0; size];
        let mut coverage = vec![0.0; size];
        let mut worth_value = vec![0.0; size];

        // Calculate accuracy parameters for all the antecedents in this rule
        let mut tn = 0.0; // True negative if use_whole
        for x in 0..size {
            let antd = &self.antds[x];
            let new_data = data.clone();
            data = Instances::new(new_data.attributes.clone());

            for ins in new_data.instances.iter() {
                if antd.covers(&ins.borrow()) {
                    // Covered by this antecedent
                    coverage[x] += 1.0;
                    data.instances.push(ins.clone()); // Add to data for further pruning
                    if ins.borrow().class_value() == self.consequent {
                        worth_value[x] += 1.0;
                    }
                } else if use_whole {
                    // Not covered
                    if ins.borrow().class_value() != self.consequent {
                        tn += 1.0;
                    }
                }
            }

            if use_whole {
                worth_value[x] += tn;
                worth_rt[x] = worth_value[x] / total as f64;
            } else {
                worth_rt[x] = (worth_value[x] + 1.0) / (coverage[x] + 2.0);
            }
        }
        dbg!(&worth_rt, &coverage, &worth_value);

        let mut max_value = (def_accu + 1.0) / (total as f64 + 2.0);
        let mut max_index = -1;
        for i in 0..worth_value.len() {
            if tracing::enabled!(Level::DEBUG) {
                let denom = if use_whole { total as f64 } else { coverage[i] };
                debug!(
                    "{}(useAccuray? {}): {}={}/{}",
                    i, !use_whole, worth_rt[i], worth_value[i], denom
                );
            }
            if worth_rt[i] > max_value {
                // Prefer to the shorter rule
                max_value = worth_rt[i];
                max_index = i as i32;
            }
        }

        // Prune the antecedents according to the accuracy parameters
        if max_index >= 0 {
            self.antds.truncate((max_index + 1) as usize);
        }
    }

    pub fn clean_up(&mut self, data: &Instances) {
        // this is only for numberic attributes, not needed for us
        // Implementation of cleanUp method
        // unimplemented!()
    }

    pub fn covers(&self, datum: &Instance) -> bool {
        self.antds.iter().all(|a| a.covers(datum))
    }

    fn compute_def_accu(&self, data: &Instances) -> f64 {
        data.instances
            .iter()
            .filter(|datum| datum.borrow().class_value() == self.consequent)
            .count() as f64
            / data.instances.len() as f64
    }

    fn compute_info_gain(
        &self,
        instances: &Instances,
        def_ac_rt: f64,
        antd: &mut NominalAntd,
    ) -> Instances {
        let instances: Vec<Instances> = antd.split_data(instances, def_ac_rt, self.consequent);
        return instances
            .get(antd.value)
            .expect("Should have this instance")
            .clone();
    }

    // Other methods...
}
