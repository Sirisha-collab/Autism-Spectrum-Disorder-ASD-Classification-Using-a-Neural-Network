"""
The Q-CHAT-10 (Quantitative Checklist for Autism in Toddlers, 10-item) used to
build the A1..A10 features from plain-language answers.

Scoring rule used by the original instrument:
  * Items 1-9  -> the three LEAST typical responses score 1, otherwise 0
  * Item 10    -> the three MOST frequent responses score 1, otherwise 0
  * Total >= 3 -> refer for a full diagnostic assessment
"""

# Each item: (feature, question, options, indices that score 1)
QCHAT10 = [
    (
        "A1",
        "Does your child look at you when you call his/her name?",
        ["Always", "Usually", "Sometimes", "Rarely", "Never"],
        {2, 3, 4},
    ),
    (
        "A2",
        "How easy is it for you to get eye contact with your child?",
        ["Very easy", "Quite easy", "Quite difficult", "Very difficult", "Impossible"],
        {2, 3, 4},
    ),
    (
        "A3",
        "Does your child point to indicate that s/he wants something "
        "(e.g. a toy that is out of reach)?",
        ["Many times a day", "A few times a day", "A few times a week",
         "Less than once a week", "Never"],
        {2, 3, 4},
    ),
    (
        "A4",
        "Does your child point to share interest with you "
        "(e.g. pointing at an interesting sight)?",
        ["Many times a day", "A few times a day", "A few times a week",
         "Less than once a week", "Never"],
        {2, 3, 4},
    ),
    (
        "A5",
        "Does your child pretend (e.g. care for dolls, talk on a toy phone)?",
        ["Many times a day", "A few times a day", "A few times a week",
         "Less than once a week", "Never"],
        {2, 3, 4},
    ),
    (
        "A6",
        "Does your child follow where you're looking?",
        ["Many times a day", "A few times a day", "A few times a week",
         "Less than once a week", "Never"],
        {2, 3, 4},
    ),
    (
        "A7",
        "If you or someone else in the family is visibly upset, does your child "
        "show signs of wanting to comfort them?",
        ["Always", "Usually", "Sometimes", "Rarely", "Never"],
        {2, 3, 4},
    ),
    (
        "A8",
        "Would you describe your child's first words as:",
        ["Very typical", "Quite typical", "Slightly unusual", "Very unusual",
         "My child doesn't speak"],
        {2, 3, 4},
    ),
    (
        "A9",
        "Does your child use simple gestures (e.g. wave goodbye)?",
        ["Many times a day", "A few times a day", "A few times a week",
         "Less than once a week", "Never"],
        {2, 3, 4},
    ),
    (
        "A10",
        "Does your child stare at nothing with no apparent purpose?",
        ["Many times a day", "A few times a day", "A few times a week",
         "Less than once a week", "Never"],
        {0, 1, 2},
    ),
]

DEMOGRAPHIC_QUESTIONS = [
    ("Age_Mons", "Child's age in months (12-36)", "int", None),
    ("Sex", "Sex of the child", "choice", ["m", "f"]),
    (
        "Ethnicity",
        "Ethnicity",
        "choice",
        ["White European", "asian", "middle eastern", "south asian", "black",
         "Hispanic", "Latino", "mixed", "Native Indian", "Pacifica", "Others"],
    ),
    ("Jaundice", "Was the child born with jaundice?", "choice", ["Yes", "No"]),
    (
        "Family_mem_with_ASD",
        "Does any immediate family member have a diagnosis of ASD?",
        "choice",
        ["Yes", "No"],
    ),
    (
        "Who completed the test",
        "Who is completing this questionnaire?",
        "choice",
        ["family member", "Health Care Professional", "Self", "Others"],
    ),
]


def score_item(feature: str, option_index: int) -> int:
    """Convert a chosen option index into the binary A-score for that item."""
    for feat, _q, options, scoring in QCHAT10:
        if feat == feature:
            if not 0 <= option_index < len(options):
                raise ValueError(f"{feature}: option index out of range")
            return 1 if option_index in scoring else 0
    raise KeyError(feature)


def score_answers(answer_indices: dict) -> dict:
    """
    answer_indices: {'A1': 0, 'A2': 3, ...} -> {'A1': 0/1, ..., 'total': int}
    """
    scores = {feat: score_item(feat, answer_indices[feat]) for feat, *_ in QCHAT10}
    scores["total"] = sum(scores.values())
    return scores
