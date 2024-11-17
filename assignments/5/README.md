<center>

# **Assignment 3 Report**
## **Name: Dikshant**
</center>


## Question 2: KDE

### 2.2
<center>
<img src="../5/figures/2.2.png"  width = 400>

dataset with two circular distribution

diffused circle = 3000 samples, 
denser circle = 500 samples 
</center>

### 2.3
<center>
<img src="../5/figures/2.3.png" width = 400>
</center>

> KDE
> - KDE successfully captured the overall shape of both circular distributions
> - The density estimate was smooth and continuous across the entire space 

> GMM
> - GMM with 2 components
>      - Captured the general structure of the two circles
>      - May have oversimplified the density variations within each circle
> - GMM with 5 components
>      - Provided a more detailed representation of density variations
>      - Risked overfitting by potentially introducing artificial substructures

KDE provided a more flexible, non-parametric representation of the data. GMM offered a parametric approach, potentially more efficient for larger datasets. KDE better captured the varying densities within each circle.

---
## Question 3
```
Trained HMM model for digit 1
Trained HMM model for digit 7
Trained HMM model for digit 3
Trained HMM model for digit 2
Trained HMM model for digit 8
Trained HMM model for digit 6
Trained HMM model for digit 5
Trained HMM model for digit 4
Trained HMM model for digit 9
Trained HMM model for digit 0
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 1
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 4
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 3
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 1
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 3
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 1
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 6
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 3, Predicted digit: 3
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 8
True digit: 6, Predicted digit: 3
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 8
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 3
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 8
True digit: 6, Predicted digit: 3
True digit: 6, Predicted digit: 8
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 3
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 5
True digit: 9, Predicted digit: 1
True digit: 9, Predicted digit: 1
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 1
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 1
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 1
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 1, Predicted digit: 1
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 1
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 7, Predicted digit: 7
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 3
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 3
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 8
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 2, Predicted digit: 2
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 1
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 1
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 1
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 1
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 1
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 3
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 5
Accuracy: 90.00%
True digit: 2, Predicted digit: 0
True digit: 2, Predicted digit: 0
True digit: 2, Predicted digit: 0
True digit: 2, Predicted digit: 0
True digit: 9, Predicted digit: 5
True digit: 9, Predicted digit: 5
True digit: 9, Predicted digit: 9
True digit: 9, Predicted digit: 9
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 5
True digit: 5, Predicted digit: 9
True digit: 5, Predicted digit: 5
True digit: 4, Predicted digit: 5
True digit: 4, Predicted digit: 4
True digit: 4, Predicted digit: 5
True digit: 4, Predicted digit: 5
True digit: 7, Predicted digit: 9
True digit: 7, Predicted digit: 9
True digit: 7, Predicted digit: 9
True digit: 7, Predicted digit: 9
True digit: 3, Predicted digit: 0
True digit: 3, Predicted digit: 9
True digit: 3, Predicted digit: 6
True digit: 3, Predicted digit: 1
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 9
True digit: 0, Predicted digit: 0
True digit: 0, Predicted digit: 0
True digit: 6, Predicted digit: 5
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 6
True digit: 6, Predicted digit: 7
True digit: 1, Predicted digit: 5
True digit: 1, Predicted digit: 5
True digit: 1, Predicted digit: 5
True digit: 1, Predicted digit: 5
True digit: 8, Predicted digit: 6
True digit: 8, Predicted digit: 6
True digit: 8, Predicted digit: 8
True digit: 8, Predicted digit: 6
Generalization Accuracy on Personal Recordings: 30.00%
```