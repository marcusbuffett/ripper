use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct NominalAttributeInfo {
    // The attribute's values
    pub values: Vec<String>,

    // Mapping of values to indices
    pub hashtable: HashMap<String, usize>,
}

impl NominalAttributeInfo {
    pub fn new(
        attribute_values: Option<Vec<String>>,
        attribute_name: &str,
    ) -> Result<Self, String> {
        match attribute_values {
            None => Ok(Self {
                values: Vec::new(),
                hashtable: HashMap::new(),
            }),
            Some(values) => {
                let mut new_values = Vec::with_capacity(values.len());
                let mut new_hashtable = HashMap::with_capacity(values.len());

                for (i, value) in values.into_iter().enumerate() {
                    if new_hashtable.contains_key(&value) {
                        return Err(format!(
                            "A nominal attribute ({}) cannot have duplicate labels ({}).",
                            attribute_name, value
                        ));
                    }

                    new_values.push(value.clone());
                    new_hashtable.insert(value, i);
                }

                Ok(Self {
                    values: new_values,
                    hashtable: new_hashtable,
                })
            }
        }
    }

    // Add some useful methods
    pub fn num_values(&self) -> usize {
        self.values.len()
    }

    // pub fn value(&self, index: usize) -> Option<&String> {
    //     self.values.get(index)
    // }
    //
    // pub fn index_of(&self, value: &str) -> Option<usize> {
    //     self.hashtable.get(value).copied()
    // }
    //
    // pub fn has_value(&self, value: &str) -> bool {
    //     self.hashtable.contains_key(value)
    // }
}
