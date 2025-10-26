# GWPCA Robusto: Fundamentos Matemáticos y Extensiones

Este documento detalla la base matemática utilizada en el paquete `gwmvt` para extender el Análisis de Componentes Principales Geográficamente Ponderado (GWPCA) clásico mediante el uso de estimadores robustos. El objetivo principal es reemplazar las estimaciones locales de media y covarianza, que son sensibles a valores atípicos, por alternativas que ofrezcan estabilidad sin perder la estructura de ponderación geográfica.

## 1. Recordatorio del GWPCA Estándar

El GWPCA estándar (Harris et al., 2011) adapta el PCA clásico al contexto espacial. Para un conjunto de datos multivariados \( x_k \in \mathbb{R}^p \) observados en ubicaciones \( s_k \in \mathbb{R}^2 \) (con \( k = 1, \dots, n \)), el GWPCA calcula una matriz de covarianza local para cada ubicación de interés \( s_i \).

El proceso es el siguiente:

1.  **Ponderación Geográfica**: Se asigna un peso \( w_k(s_i) \) a cada observación \( x_k \) en función de su distancia a \( s_i \). Estos pesos se calculan mediante una función kernel (ej. Gaussiano).

    \begin{equation}
    w_k(s_i) = \exp\left(-\frac{d(s_i, s_k)^2}{2h^2}\right)
    \end{equation}

2.  **Estimación Local**: Se calculan la media y la matriz de covarianza ponderadas.

    *   **Media Local Ponderada**:
        \begin{equation}
        \bar{x}(s_i) = \frac{\sum_{k=1}^{n} w_k(s_i)x_{k}}{\sum_{k=1}^{n} w_k(s_i)}
        \end{equation}

    *   **Matriz de Covarianza Local Ponderada**:
        \begin{equation}
        \Sigma(s_i) = \frac{\sum_{k=1}^{n} w_k(s_i) (x_k - \bar{x}(s_i))(x_k - \bar{x}(s_i))^T}{\sum_{k=1}^{n} w_k(s_i)}
        \end{equation}

3.  **Extracción de Componentes**: Se realiza una descomposición de valores propios sobre \( \Sigma(s_i) \) para obtener los componentes principales locales.

    \begin{equation}
    \Sigma(s_i) \phi_m(s_i) = \lambda_m(s_i) \phi_m(s_i)
    \end{equation}

## 2. La Filosofía del GWPCA Robusto

El GWPCA robusto sustituye los estimadores locales de media y covarianza por alternativas robustas.

\begin{equation}
(\tilde{\bar{x}}(s_i), \tilde{\Sigma}(s_i)) = \mathcal{R} \Bigl(\bigl\{ (x_k, w_k(s_i)) \bigr\}_{k=1}^n \Bigr)
\end{equation}

Donde \( \mathcal{R} \) es un algoritmo de estimación robusto. A continuación, se describen los estimadores implementados en `/src/methods/pca/robust/`.

## 3. Estimadores Robustos Implementados

### 3.1. M-Estimador de Huber

*   **Concepto**: (Huber, 1964). Limita la influencia de los outliers usando una función de pérdida que crece más lentamente que la cuadrática para residuos grandes.

*   **Matemáticas**: La función de pérdida de Huber es:
    \begin{equation}
    \rho_c(u) =
    \begin{cases}
    \frac{1}{2}u^2, & |u| \le c, \
    c |u| - \frac{1}{2} c^2, & |u| > c.
    \end{cases}
    \end{equation}

*   **Implementación (Algoritmo de Maronna, 1976)**: Se resuelven iterativamente la media y covarianza mediante un enfoque de re-ponderación (IRWLS) hasta la convergencia.

### 3.2. Determinante de Covarianza Mínima (MCD)

*   **Concepto**: (Rousseeuw, 1984). Estimador de alto punto de ruptura que busca el subconjunto de \( h \) observaciones cuya covarianza clásica tenga el menor determinante posible.

*   **Implementación (Fast-MCD)**: (Rousseeuw & Van Driessen, 1999). Utiliza "C-Pasos" para encontrar eficientemente el mejor subconjunto. Finaliza con un paso de re-ponderación basado en distancias de Mahalanobis para mejorar la eficiencia.

### 3.3. Elipsoide de Volumen Mínimo (MVE)

*   **Concepto**: (Rousseeuw, 1985). Busca el elipsoide de mínimo volumen que contenga al menos \( h \) de las observaciones. El centro y la forma de este elipsoide definen la media y covarianza robustas.

### 3.4. S-Estimadores

*   **Concepto**: (Rousseeuw & Yohai, 1984). Minimizan una M-estimación de la escala de los residuos. Son de alto punto de ruptura.

*   **Matemáticas**: Minimizan una escala \( s \) definida implícitamente por:
    \begin{equation}
    \frac{1}{n} \sum_{k=1}^n \rho\left(\frac{d_k}{s}\right) = K
    \end{equation}
    donde \( d_k \) son las distancias de Mahalanobis y \( \rho \) es una función de pérdida (e.g., biponderada de Tukey).

### 3.5. MM-Estimadores

*   **Concepto**: (Yohai, 1987). Combinan la alta robustez de los S-estimadores con la alta eficiencia de los M-estimadores en un proceso de dos etapas.

### 3.6. BACON (Blocked Adaptive Computationally-efficient Outlier Nominators)

*   **Concepto**: (Billor et al., 2000). Algoritmo iterativo que identifica un subconjunto "limpio" haciéndolo crecer desde un núcleo seguro de puntos no atípicos.

### 3.7. Ponderación por Factor Local de Outlier (LOF)

*   **Concepto**: (Breunig et al., 2000). No es un estimador de covarianza, sino un método para detectar outliers basado en la densidad local. Un punto con LOF alto está en una región de densidad mucho menor que la de sus vecinos.

*   **Matemáticas**: El LOF de un punto \(x_k\) se define como el promedio de la relación entre la densidad local de sus vecinos y su propia densidad local.
    \begin{equation}
    LOF(x_k) = \frac{1}{|N_j|} \sum_{j \in N_k} \frac{lrd(x_j)}{lrd(x_k)}
    \end{equation}
    donde \(lrd\) es la densidad de alcanzabilidad local (*local reachability density*) y \(N_k\) es el conjunto de vecinos de \(x_k\).

*   **Implementación**: Se calcula el LOF para cada punto. Luego, se define un peso de robustez \(w_{rob,k} = 1 / LOF(x_k)\). La media y covarianza se calculan usando los pesos geográficos \(w_k(s_i)\) multiplicados por estos pesos de robustez.

### 3.8. ROBPCA

*   **Concepto**: (Hubert et al., 2005). Combina la robustez de MCD con una técnica de reducción de dimensionalidad (Proyección-Persecución). Es ideal para datos de alta dimensionalidad.

*   **Implementación**: 
    1.  El algoritmo primero proyecta los datos a un subespacio de menor dimensión \(q < p\) donde se concentra la mayor parte de la variabilidad.
    2.  En este subespacio, aplica un estimador robusto como MCD para obtener una media y covarianza robustas.
    3.  Finalmente, retro-proyecta estas estimaciones al espacio original para obtener la matriz de covarianza robusta final \( \tilde{\Sigma} \).

### 3.9. Estimador Basado en Profundidad Espacial (Spatial Depth)

*   **Concepto**: La profundidad de un dato (Vardi & Zhang, 2000) mide su centralidad respecto a la nube de puntos. Los outliers son puntos con baja profundidad.

*   **Matemáticas**: La profundidad espacial de un punto \(y\) respecto a un conjunto de datos \(X = \{x_k\}_{k=1}^n\) es:
    \begin{equation}
    SD(y, X) = 1 - \left\| \frac{1}{n} \sum_{k=1}^n \frac{y - x_k}{\|y - x_k\|} \right\|
    \end{equation}

*   **Implementación**: Se calcula la profundidad \(d_k = SD(x_k, X)\) para cada punto. Estos valores de profundidad se usan como pesos de robustez, que se multiplican por los pesos geográficos para calcular una media y covarianza ponderadas.

### 3.10. Recorte Espacial (Spatial Trimming)

*   **Concepto**: Es un método general (Maronna et al., 2019) que consiste en "recortar" o eliminar las observaciones que se identifican como outliers antes de calcular un estimador clásico.

*   **Implementación**:
    1.  Se parte de una estimación inicial robusta de media y covarianza (e.g., MCD).
    2.  Se calculan las distancias de Mahalanobis robustas \(d_R(k)\) para todos los puntos.
    3.  Se define un conjunto "limpio" \(H\) con los puntos cuya distancia al cuadrado no supera un umbral de la distribución \(\chi^2_p\).
        \begin{equation}
        H = \{k \mid d_R(k)^2 \le \chi^2_{p, 1-\alpha}\}
        \end{equation}
    4.  La media y covarianza robustas finales son la media y covarianza ponderadas estándar, calculadas únicamente sobre el subconjunto \(H\).

## 4. Regularización y Estabilidad Numérica

Independientemente del estimador, la matriz \( \tilde{\Sigma} \) debe ser simétrica y definida positiva. El código implementa salvaguardas como la **simetrización** (\( \tilde{\Sigma} \leftarrow \frac{1}{2}(\tilde{\Sigma} + \tilde{\Sigma}^T) \)) y la **regularización Ridge** (sumar \( \lambda \mathbf{I} \) a la diagonal).

## 5. Referencias Bibliográficas

-   Billor, N., Hadi, A. S., & Velleman, P. F. (2000). BACON: Blocked Adaptive Computationally-efficient Outlier Nominators. *Computational Statistics & Data Analysis*, 34(3), 279-298.
-   Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: identifying density-based local outliers. In *Proceedings of the 2000 ACM SIGMOD international conference on Management of data* (pp. 93-104).
-   Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002). *Geographically weighted regression: the analysis of spatially varying relationships*. John Wiley & Sons.
-   Harris, P., Clarke, A., Juggins, S., Brunsdon, C., & Charlton, M. (2011). Geographically weighted principal components analysis. *Geographical and Environmental Modelling*, 15(1), 29-52.
-   Huber, P. J. (1964). Robust Estimation of a Location Parameter. *The Annals of Mathematical Statistics*, 35(1), 73-101.
-   Hubert, M., Rousseeuw, P. J., & Vanden Branden, K. (2005). ROBPCA: a new approach to robust principal component analysis. *Technometrics*, 47(1), 1-12.
-   Maronna, R. A. (1976). Robust M-estimators of multivariate location and scatter. *The Annals of Statistics*, 4(1), 51-67.
-   Maronna, R. A., Martin, R. D., & Yohai, V. J. (2019). *Robust statistics: theory and methods (with R)*. John Wiley & Sons.
-   Rousseeuw, P. J. (1984). Least Median of Squares Regression. *Journal of the American Statistical Association*, 79(388), 871-880.
-   Rousseeuw, P. J. (1985). Multivariate Estimation with High Breakdown Point. In *Mathematical Statistics and Applications* (pp. 283-297). Springer, Dordrecht.
-   Rousseeuw, P. J., & Van Driessen, K. (1999). A fast algorithm for the minimum covariance determinant estimator. *Technometrics*, 41(3), 212-223.
-   Rousseeuw, P. J., & Yohai, V. J. (1984). Robust regression by means of S-estimators. In *Robust and nonlinear time series analysis* (pp. 256-272). Springer, New York, NY.
-   Vardi, Y., & Zhang, C. H. (2000). The multivariate L1-median and associated data depth. *Proceedings of the National Academy of Sciences*, 97(4), 1423-1426.
-   Yohai, V. J. (1987). High breakdown-point and high efficiency robust estimates for regression. *The Annals of Statistics*, 15(2), 642-656.
