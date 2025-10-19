**Things I accomplished while making this model:**

- Built your first ML model
- Understand train/test split and why it matters
- Know the difference between simple models (Logistic Regression) and complex ones (Random Forest)
- Seeing overfitting in action
- Handling real-world data issues (nulls, categorical encoding)


## Additional Feature Engineering Experiments

Tried adding:
- `FamilySize = SibSp + Parch` → No improvement (redundant with existing features)
- `IsAlone` binary flag → Accuracy dropped to 78%
- `Title` from passenger name → Accuracy dropped to 78%

**Conclusion:** Original 6 features (Pclass, Sex, Age, SibSp, Parch, Fare) 
performed best at 81% accuracy. Sometimes simpler is better.