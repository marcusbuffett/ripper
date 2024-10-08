use crate::attribute_info::NominalAttributeInfo;

#[derive(Clone, Debug)]
pub struct Attribute {
    pub name: String,
    pub index: usize,
    pub attribute_info: Option<NominalAttributeInfo>,
}

impl Attribute {
    pub fn num_values(&self) -> usize {
        self.attribute_info
            .as_ref()
            .expect("Should have attribute info")
            .num_values()
    }

    pub(crate) fn index_of_value(&self, value: &str) -> Option<usize> {
        let attribute_info = self
            .attribute_info
            .as_ref()
            .expect("Should have attribute info");
        attribute_info.hashtable.get(value).copied()
    }

    pub(crate) fn force_add_value(&mut self, value: &str) -> () {
        let attribute_info = self
            .attribute_info
            .as_mut()
            .expect("Should have attribute info");
        attribute_info.values.push(value.to_string());
        attribute_info
            .hashtable
            .insert(value.to_string(), attribute_info.num_values() - 1);
    }
}
