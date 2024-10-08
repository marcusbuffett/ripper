// rule_set.rs

use crate::rule::Rule;

#[derive(Clone, Default)]
pub struct RuleSet {
    pub rules: Vec<Rule>,
    // pub extension: Vec<Extension>,
    // pub rule_selection_method: Vec<RuleSelectionMethod>,
    // pub score_distribution: Vec<ScoreDistribution>,
    // pub rule: Vec<Rule>,
    // pub default_confidence: Option<f64>,
    // pub default_score: Option<String>,
    // pub nb_correct: Option<f64>,
    // pub record_count: Option<f64>,
}

impl RuleSet {
    pub fn new() -> Self {
        Default::default()
    }
}

impl std::fmt::Debug for RuleSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RuleSet {{ rules: {:?} }}", self.rules)
    }
}
