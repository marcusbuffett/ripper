use itertools::Itertools;

use crate::{
    attribute::Attribute,
    instance::{Instance, Instances},
};

#[derive(Debug, Clone)]
pub struct NominalAntd {
    pub attr: Attribute,
    pub value: usize,
    pub max_info_gain: f64,
    pub accu_rate: f64,
    pub cover: f64,
    pub accu: f64,
    pub accurate: Vec<f64>,
    pub coverage: Vec<f64>,
}

impl NominalAntd {
    pub fn new(attr: Attribute) -> Self {
        let bag = attr.num_values();

        let mut n_antd = NominalAntd {
            attr,
            value: 0,
            max_info_gain: 0.0,
            accu_rate: f64::NAN,
            cover: f64::NAN,
            accu: f64::NAN,
            accurate: Vec::new(),
            coverage: Vec::new(),
        };

        let accurate = Vec::with_capacity(bag);
        let coverage = Vec::with_capacity(bag);
        n_antd.accurate = accurate;
        n_antd.coverage = coverage;
        n_antd
    }

    pub fn split_data(&mut self, data: &Instances, def_ac_rt: f64, cl: usize) -> Vec<Instances> {
        let bag = self.attr.num_values();
        let mut split_data = (0..bag)
            .map(|_| Instances::new(data.attributes.clone()))
            .collect_vec();

        self.accurate = vec![0.0; bag];
        self.coverage = vec![0.0; bag];

        for inst in data.instances.iter() {
            if !inst.borrow().is_missing(&self.attr) {
                let v = inst.borrow().value(&self.attr);
                split_data[v].instances.push(inst.clone());
                self.coverage[v] += 1.0;
                if inst.borrow().class_value() == cl {
                    self.accurate[v] += 1.0;
                }
            }
        }

        for x in 0..bag {
            let t = self.coverage[x] + 1.0;
            let p = self.accurate[x] + 1.0;
            let info_gain = self.accurate[x] * ((p / t).log2() - (def_ac_rt).log2());

            if info_gain > self.max_info_gain {
                self.max_info_gain = info_gain;
                self.cover = self.coverage[x];
                self.accu = self.accurate[x];
                self.accu_rate = p / t;
                self.value = x;
            }
        }

        split_data
    }

    pub fn covers(&self, inst: &Instance) -> bool {
        inst.value(&self.attr) == self.value
    }
}
