use itertools::Itertools;
use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use crate::attribute::Attribute;

#[derive(Clone, Debug)]
pub struct Instances {
    pub instances: Vec<RefCell<Instance>>,
    pub attributes: RefCell<Vec<Attribute>>,
}

impl Instances {
    pub fn new(attributes: RefCell<Vec<Attribute>>) -> Self {
        Instances {
            instances: vec![],
            attributes: attributes,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Instance {
    // index of the attribute value, since strings are slow
    pub attribute_values: Vec<usize>,
    // pub weight: f64,
    // pub numeric_after_decimal_point: i32,
    pub dataset: Option<RefCell<Instances>>,
}

impl Instance {
    pub fn class_value(&self) -> usize {
        *self.attribute_values.last().expect("No class value")
    }

    pub fn get_attribute(&self, index: usize) -> usize {
        *self
            .attribute_values
            .get(index)
            .expect("No attribute value")
    }

    pub fn is_missing(&self, attr: &crate::attribute::Attribute) -> bool {
        self.attribute_values.get(attr.index).is_none()
    }

    pub(crate) fn value(&self, attr: &crate::attribute::Attribute) -> usize {
        *(self
            .attribute_values
            .get(attr.index)
            .expect("Attribute index out of bounds"))
    }

    pub fn set_value(&mut self, att_idx: usize, value: String) {
        let mut dataset = self
            .dataset
            .as_mut()
            .expect("Should have a dataset")
            .borrow_mut();
        let mut attributes = dataset.attributes.borrow_mut();
        let attr = attributes
            .get_mut(att_idx)
            .expect("Should have this attribute");
        let val_idx = match attr.index_of_value(&value) {
            Some(i) => i,
            None => {
                attr.force_add_value(&value);
                attr.index_of_value(&value)
                    .expect("This should be impossible")
            }
        };
        self.attribute_values[val_idx] = val_idx;
    }
}

impl Instances {
    pub fn by_class_frequency() -> Self {
        unimplemented!()
    }

    pub fn sort_by_class_freq_asc(&mut self) {
        let class_frequencies = self.class_frequencies();
        self.instances.sort_by(|a, b| {
            let a_freq = class_frequencies
                .get(&a.borrow().class_value())
                .unwrap_or(&0);
            let b_freq = class_frequencies
                .get(&b.borrow().class_value())
                .unwrap_or(&0);
            a_freq.cmp(b_freq)
        });
    }

    pub fn class_frequencies(&self) -> HashMap<usize, usize> {
        let class_frequencies = self
            .instances
            .iter()
            .map(|i| (*i).borrow().class_value())
            .counts()
            .into_iter()
            .map(|(k, v)| (k.clone(), v))
            .collect::<HashMap<usize, usize>>();
        class_frequencies
    }

    pub fn ordered_class_frequencies(&self) -> Vec<(usize, usize)> {
        let class_frequencies = self.class_frequencies();
        class_frequencies
            .into_iter()
            .sorted_by_key(|(_, v)| *v)
            .collect_vec()
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

    pub fn num_classes(&self) -> usize {
        self.attributes
            .borrow()
            .last()
            .expect("No attributes")
            .attribute_info
            .as_ref()
            .expect("No attribute info")
            .num_values()
    }
}
