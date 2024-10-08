// rule_set.rs

#[derive(Debug, Clone, Default)]
pub struct RuleSet {
    pub extension: Vec<Extension>,
    pub rule_selection_method: Vec<RuleSelectionMethod>,
    pub score_distribution: Vec<ScoreDistribution>,
    pub rule: Vec<Rule>,
    pub default_confidence: Option<f64>,
    pub default_score: Option<String>,
    pub nb_correct: Option<f64>,
    pub record_count: Option<f64>,
}

impl RuleSet {
    pub fn new() -> Self {
        Default::default()
    }
}

#[derive(Debug, Clone)]
pub struct Extension {
    // Add fields as needed
}

#[derive(Debug, Clone)]
pub struct RuleSelectionMethod {
    // Add fields as needed
}

#[derive(Debug, Clone)]
pub struct ScoreDistribution {
    // Add fields as needed
}

#[derive(Debug, Clone)]
pub enum Rule {
    SimpleRule(SimpleRule),
    CompoundRule(CompoundRule),
}

#[derive(Debug, Clone)]
pub struct SimpleRule {
    // Add fields as needed
}

#[derive(Debug, Clone)]
pub struct CompoundRule {
    // Add fields as needed
}
