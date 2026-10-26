# Formative 3: Probability Distributions, Bayes' Theorem, and Gradient Descent

**Group:** 12  

**Course:** Mathematics for Machine Learning  

**Institution:** African Leadership University

## 1. Project Overview

This project explores the application of probability, statistical learning, and optimization techniques through four practical tasks. The implementation covers the Expectation-Maximization (EM) algorithm for Gaussian mixture models, Bayes' theorem for IMDb movie reviews, manual gradient descent calculations, and gradient descent implemented in Python using SciPy.

The project combines mathematical derivations, Python implementations, numerical results, and visualizations to demonstrate how these techniques work and how they can be applied to data.

## 2. Project Objectives

The main objectives are to:

- Implement the Expectation-Maximization algorithm to estimate the parameters of a two-component Gaussian mixture.

- Apply Bayes' theorem to calculate the probability that an IMDb review is positive given a keyword.

- Derive and perform gradient descent updates manually using matrix operations and the chain rule.

- Implement gradient descent in Python using SciPy to calculate numerical derivatives.

- Use Matplotlib to visualize parameter changes and the reduction in prediction error.

- Compare manual calculations with Python results to check the correctness of the implementation.

## 3. Technologies and Tools

- **Python:** Main programming language.

- **Jupyter Notebook:** Running code, presenting calculations, and displaying results.

- **NumPy:** Matrix operations and numerical calculations.

- **SciPy:** Numerical differentiation for gradient descent.

- **Matplotlib:** Visualizing Gaussian mixture results, posterior probabilities, parameter changes, and error reduction.

- **Git and GitHub:** Version control and collaborative development.

The Bayes' theorem implementation uses Python's standard `csv` and `re` libraries for reading and processing review data.

## 4. Project Structure

The repository is organized into separate modules to keep the implementation clear and maintainable.

| File | Purpose |

|---|---|

| `Group 12 Formative 3 Notebook.ipynb` | Main notebook containing explanations, experiments, results, and visualizations for Parts 1–4. |

| `em.py` | Implements the Gaussian mixture model and Expectation-Maximization algorithm. |

| `bayes.py` | Loads IMDb reviews, calculates keyword probabilities, and formats the results. |

| `gradient_descent.py` | Implements predictions, MSE, analytical gradients, numerical gradients, parameter updates, and the derivative function. |

| `plots.py` | Contains functions for plotting the Gaussian mixture, posterior probabilities, parameter changes, and error. |

| `utils.py` | Provides helper functions for loading and preparing the height data and displaying the EM tracking table. |

| `GaltonFamilies.csv` | Dataset used for the height-distribution analysis. |

| `imdb.csv` | Dataset used for the IMDb keyword-probability analysis. |

| `part4_params.png` | Saved visualization of parameter changes during gradient descent, when generated. |

| `part4_error.png` | Saved visualization of the error during gradient descent, when generated. |

The handwritten gradient descent calculations and the group contribution document are supporting submission materials.

**Note: The file names above should match the actual names in the GitHub repository. If a file has a different name, update this table accordingly.**

## 5. Part 1: Expectation-Maximization for a Gaussian Mixture Model

### Objective

To fit a mixture of two Gaussian distributions to an unlabeled collection of heights and estimate the probability that a given height belongs to each component.

### Dataset

The analysis uses `GaltonFamilies.csv`, containing fathers' and children's heights. The data is prepared so that each father is represented once. Fathers' and children's heights are then pooled into one collection of 1,139 observations, and the model is fitted without using the labels.

### Implementation

The EM algorithm is implemented in `em.py` and follows these steps:

1. **Initialization:** Initialize the component means using the 25th and 75th percentiles, initialize both variances using the data variance, and assign equal mixing coefficients.

2. **Expectation step (E-step):** Calculate the responsibility of each Gaussian component for every observation.

3. **Maximization step (M-step):** Update the means, variances, and mixing coefficients using the calculated responsibilities.

4. **Convergence:** Repeat the E-step and M-step until the change in log-likelihood is smaller than the specified tolerance.

### Analysis and outputs

The notebook includes:

- A tracking table showing the initial estimates and the first two iterations (iterations 0, 1, and 2).

- The final estimated means, variances, and mixing coefficients.

- A comparison between the EM approach and a simple split at the global mean.

- A visualization of the fitted Gaussian components.

- Posterior probabilities for a test height provided to the classifier.

The test height can be supplied in inches or centimeters. The model converts centimeters to inches before calculating the posterior probabilities.

**Interpretation:** The two Gaussian components represent probabilistic clusters based on height. They should not automatically be interpreted as fathers and children or as definitive demographic categories.

## 6. Part 2: Bayes' Theorem on IMDb Reviews

### Objective

To apply Bayes' theorem to estimate the probability that a movie review is positive given that a selected keyword appears in it.

### Selected keywords

**Positive keywords:**

- excellent

- wonderful

- amazing

- superb

**Negative keywords:**

- awful

- waste

- boring

- worst

The implementation calculates only P(Positive|keyword), as selected for this assignment.

### Mathematical approach

Bayes' theorem is expressed as:

P(Positive | w) = [P(w | Positive) × P(Positive)] / P(w)

Where:

- **Prior:** The probability that a randomly selected review is positive.

- **Likelihood:** The probability that a positive review contains the selected keyword.

- **Marginal probability:** The probability that a review contains the selected keyword.

- **Posterior:** The probability that a review is positive given that it contains the keyword.

### Implementation and outputs

The implementation in `bayes.py` loads the reviews from `imdb.csv`, counts keyword occurrences, calculates the required probabilities, and displays the results in a table.

The notebook also checks that the posterior calculated using Bayes' theorem agrees with the directly counted conditional probability.

A keyword is considered present if it appears at least once in a review. Text is converted to lowercase, and HTML line-break tags are removed during preprocessing.

## 7. Part 3: Manual Gradient Descent Calculations

### Objective

To understand gradient descent by deriving the gradients and manually performing three parameter updates.

### Initial values

The model uses the matrix equation:

ŷ = Xm + b

With:

X = [[1, 3], [4, 10]],    y = [[5], [6]]

m₀ = [[−1], [2]],    b₀ = [[1], [1]]

The learning rate is α = 0.01, and the number of observations is n=2.

### Cost function and gradients

The Mean Squared Error is:

J = (1/n)(ŷ − y)ᵀ(ŷ − y)

Let e = ŷ − y. The gradients are:

∇ₘJ = (2/n)Xᵀe

∇ᵦJ = (2/n)e

The parameters are updated using:

m_new = m_old − α∇ₘJ

b_new = b_old − α∇ᵦJ

Since n=2, the factor 2/n equals 1.

### Expected results

The initial predictions are [6, 17]ᵀ, with an initial MSE of 61. The notebook performs three updates and records the changes in the parameters, predictions, errors, and MSE.

The handwritten calculations provide the detailed mathematical steps, while the notebook provides the corresponding Python implementation for comparison.

## 8. Part 4: Gradient Descent in Python Using SciPy

### Objective

To implement gradient descent in Python, calculate derivatives numerically using SciPy, and visualize how the parameters and error change during optimization.

### Implementation

The implementation in `gradient_descent.py` includes:

- A function that accepts a single-variable equation and computes its derivative using SciPy.

- A function that calculates numerical gradients for the model parameters.

- Analytical gradients derived using the chain rule and matrix operations.

- A gradient descent loop that updates m and b, calculates predictions, and evaluates the MSE.

The notebook compares two runs:

1. Gradient descent using numerical gradients calculated with SciPy.

2. Gradient descent using analytical gradients calculated from the chain rule.

Both implementations use the same initial values and learning rate, allowing their results to be compared.

### Visualizations

The notebook generates the following plots:

- **Parameter changes:** Shows how the components of m and b change during gradient descent.

- **Error reduction:** Shows how the MSE changes over the iterations.

The plots are saved as `part4_params.png` and `part4_error.png`, respectively, when the relevant notebook cells are executed.

### Expected outcome

The initial MSE is 61. After 300 iterations, the notebook reports an MSE of approximately 0.004522 and predictions close to the target values [5, 6]ᵀ.

The notebook also checks whether the numerical-gradient and analytical-gradient implementations produce matching results.

## 9. How to Run the Project

### Prerequisites

Install Python and Jupyter Notebook or Google Collab. The project requires NumPy, SciPy, and Matplotlib for its numerical calculations and visualizations.

### Installation

Install the required packages:

```bash

python -m pip install numpy scipy matplotlib notebook

```

### Setup

1. Clone the Group 12 GitHub repository or download the project files.

2. Open the project folder in a terminal or code editor.

3. Ensure that the Python modules and required CSV datasets are available in the project folder, or update the dataset paths if the repository uses subfolders.

4. Launch Jupyter Notebook:

```bash

jupyter notebook

```

5\. Open `Group 12 Formative 3 Notebook.ipynb`.

6\. Run the notebook cells from top to bottom.

Run the notebook from the project directory so that its imports and dataset paths resolve correctly.

**Important:** The project also requires the SciPy version to support the `scipy.differentiate.derivative` function used in `gradient_descent.py`. If this import is unavailable in your installed version, use a compatible SciPy version before running Part 4.

## 10. Validation and Testing

The notebook includes checks to support the correctness of the implementation:

- The EM model tracks log-likelihood values across iterations.

- Posterior probabilities are calculated for the selected test height.

- The Bayes' theorem posterior is checked against the directly counted conditional probability.

- The analytical gradients are compared with the numerical gradients calculated using SciPy.

- The first three gradient descent updates are compared with the manual calculations.

- The final parameter values, predictions, and MSE are displayed after 300 iterations.

These checks help identify differences between the mathematical calculations and the code output.

## 11. Submission Deliverables

The complete submission should include the following items, as required by the assignment:

- **README.md:** Project overview, methods, repository structure, and instructions for running the code.

- **Jupyter Notebook:** Implementations, calculations, explanations, results, and visualizations for Parts 1–4.

- **Python modules:** The source code used for EM, Bayes' theorem, gradient descent, plotting, and helper functions.

- **Dataset files:** The CSV files required to reproduce the results, subject to the dataset's sharing and licensing requirements.

- **Handwritten Part 3 calculations:** A neat, combined PDF showing the manual gradient descent calculations.

- **Group contribution document:** A PDF documenting each group member's contribution.

- **GitHub commit history:** Evidence of collaborative work and changes made to the project.

- **Group presentation:** A 15-minute presentation explaining the methods, implementation, results, and key observations.

Ensure that all required files are present in the submission location specified by the instructor and that the notebook can be executed successfully.

## 12. Conclusion

This project demonstrates how probability and optimization methods can be implemented and evaluated using Python. The EM algorithm estimates the parameters of a Gaussian mixture from unlabeled height data, Bayes' theorem evaluates the relationship between IMDb review keywords and positive sentiment, and gradient descent illustrates how model parameters can be updated to reduce prediction error.

By combining manual derivations with Python implementations and visualizations, the project connects the mathematical concepts to practical computation and emphasizes the importance of testing and validating results.
