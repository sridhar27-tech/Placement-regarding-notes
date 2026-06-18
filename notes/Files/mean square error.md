# Mean Squared Error (MSE)

**Mean Squared Error (MSE)** is one of the most commonly used **loss functions** in regression problems. It measures the average squared difference between the predicted values and the actual (true) values.

---

## Definition

MSE quantifies how well a model's predictions match the actual data. **Lower MSE = Better Model**.

It penalizes larger errors more heavily because the differences are squared.

---

## Mathematical Formula

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where:
- $n$ = number of data points (samples)
- $y_i$ = actual (true) value
- $\hat{y}_i$ = predicted value

---

## Root Mean Squared Error (RMSE)

Often used alongside MSE for better interpretability (same unit as the target variable):

$$
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

---

## Properties of MSE

- **Always non-negative**
- **Differentiable** → Easy to optimize using Gradient Descent
- **Convex** when used with linear models
- **Sensitive to outliers** (because errors are squared)

---

## When to Use MSE?

- **Best for**: Regression tasks where you want to penalize large errors more (e.g., house price prediction, temperature forecasting).
- Commonly used in:
  - Linear Regression
  - Neural Networks
  - Time Series Forecasting

---

## Advantages

- Simple and easy to understand
- Smooth and differentiable (good for optimization)
- Gives higher weight to large errors

---

## Disadvantages

- Sensitive to outliers (one bad prediction can significantly increase the loss)
- Not suitable for classification problems
- Units are squared (harder to interpret) → hence RMSE is often preferred

---

## Comparison with Other Loss Functions

| Loss Function       | Best For              | Sensitive to Outliers | Differentiable |
|---------------------|-----------------------|-----------------------|----------------|
| **MSE**             | Regression            | High                  | Yes            |
| MAE (Mean Absolute Error) | Regression       | Low                   | No             |
| Huber Loss          | Robust Regression     | Medium                | Yes            |
| Cross-Entropy       | Classification        | -                     | Yes            |

---

**Tip**: When training models with Gradient Descent, MSE is often the default choice for regression problems because of its smooth gradient.
Would you like the explanation with **numerical examples**, **code snippets** (Python), or comparison with other loss functions?

Mathematical Calcualtions
# Mean Squared Error (MSE) - Mathematical Calculations & Example

## 1. Mathematical Formula

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where:
- $n$ = Total number of data points
- $y_i$ = Actual (true) target value for the $i$-th sample
- $\hat{y}_i$ = Predicted value for the $i$-th sample

---

## 2. Step-by-Step Mathematical Calculation

**Step 1**: Calculate the **error** for each data point  
$$
\text{Error}_i = y_i - \hat{y}_i
$$

**Step 2**: Square each error (to remove negative signs and penalize large errors more)  
$$
\text{Squared Error}_i = (y_i - \hat{y}_i)^2
$$

**Step 3**: Sum all squared errors  
$$
\text{Sum of Squared Errors (SSE)} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**Step 4**: Take the mean (average)  
$$
\text{MSE} = \frac{\text{SSE}}{n} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

---

## 3. Prioritized Numerical Example

**Problem**:  
We have a simple house price prediction model. Here are 5 actual vs predicted prices (in lakhs):

| Sample | Actual Price ($y_i$) | Predicted Price ($\hat{y}_i$) |
|--------|----------------------|-------------------------------|
| 1      | 25                   | 23                            |
| 2      | 30                   | 32                            |
| 3      | 40                   | 35                            |
| 4      | 50                   | 55                            |
| 5      | 60                   | 58                            |

---

### Step-by-Step Calculation:

**Step 1: Calculate Errors**  
- Sample 1: $25 - 23 = 2$  
- Sample 2: $30 - 32 = -2$  
- Sample 3: $40 - 35 = 5$  
- Sample 4: $50 - 55 = -5$  
- Sample 5: $60 - 58 = 2$

**Step 2: Square the Errors**  
- Sample 1: $2^2 = 4$  
- Sample 2: $(-2)^2 = 4$  
- Sample 3: $5^2 = 25$  
- Sample 4: $(-5)^2 = 25$  
- Sample 5: $2^2 = 4$

**Step 3: Sum of Squared Errors (SSE)**  
$$
4 + 4 + 25 + 25 + 4 = 62
$$

**Step 4: Calculate MSE**  
$$
\text{MSE} = \frac{62}{5} = 12.4
$$

---

### Final Results:

- **MSE = 12.4**
- **RMSE** (Root Mean Squared Error) = $\sqrt{12.4} \approx 3.52$ (in lakhs)

**Interpretation**:  
On average, the model's predictions are off by about **3.52 lakhs** from the actual prices.

---

## Why This Example is Prioritized?

- Small dataset → Easy to follow manually
- Shows both positive and negative errors
- Demonstrates how large errors (5 and -5) contribute more due to squaring
- Practical regression use case (house price prediction)

---

**Key Takeaways**:
- MSE is **differentiable** → Perfect for Gradient Descent
- It is **sensitive to outliers** (large errors dominate the loss)
- Used as a loss function during training and as an evaluation metric

Would you like:
- More examples (with bigger datasets)?
- Python code to compute MSE?
- Comparison with MAE using the same example?

Just let me know!
