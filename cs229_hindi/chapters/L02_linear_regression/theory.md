# Supervised Learning Setup और Linear Regression

> **इस lecture में:** Supervised Learning का पूरा ढाँचा (training set, hypothesis, cost function), Linear Regression, Gradient Descent (Batch और Stochastic), Normal Equations, और यह कि Least Squares असल में Maximum Likelihood क्यों है।

## 1. एक line में idea

हमारे पास कुछ उदाहरण हैं जिनमें input $x$ और सही जवाब $y$ दोनों दिए हैं। हमें ऐसा function $h$ सीखना है जो नए $x$ के लिए $y$ का अच्छा अंदाज़ा लगा सके। जब $y$ एक continuous संख्या हो (जैसे घर की कीमत), तो इस problem को **regression** कहते हैं। जब $y$ कुछ गिनी-चुनी categories में से एक हो (जैसे spam / not spam), तो इसे **classification** कहते हैं।

Linear Regression सबसे सरल regression model है: हम मान लेते हैं कि $y$ लगभग $x$ के features का एक weighted sum है, और फिर data से सबसे अच्छे weights ढूँढते हैं।

## 2. Setup और Notation

| Symbol | मतलब |
|---|---|
| $x^{(i)}$ | $i$-वें training example के input features (एक vector) |
| $y^{(i)}$ | $i$-वें example का target (सही जवाब) |
| $(x^{(i)}, y^{(i)})$ | एक training example |
| $m$ | training examples की संख्या |
| $n$ | features की संख्या |
| $\theta$ | parameters (weights), जिन्हें हमें सीखना है |
| $h_\theta(x)$ | hypothesis, यानी model का prediction |

Linear Regression में hypothesis यह है (हम $x_0 = 1$ मान लेते हैं ताकि intercept $\theta_0$ भी उसी formula में आ जाए):

$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n = \sum_{j=0}^{n} \theta_j x_j = \theta^T x$$

एक feature हो तो यह एक line है, दो features हों तो एक plane, और ज़्यादा हों तो hyperplane।

## 3. Cost Function

$\theta$ कितना अच्छा है, यह नापने के लिए हम देखते हैं कि predictions असली $y$ से कितनी दूर हैं। इसी दूरी के squares का जोड़ **cost function** है:

$$J(\theta) = \frac{1}{2} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2$$

$J(\theta)$ जितना छोटा, fit उतना अच्छा। Linear Regression के लिए $J$ एक **convex** कटोरे (bowl) जैसा होता है, इसलिए इसका सिर्फ़ एक ही global minimum है और कोई local minimum नहीं।

## 4. Gradient Descent

कटोरे की तली तक पहुँचने का तरीका: जहाँ खड़े हैं वहाँ से सबसे तेज़ ढलान की दिशा में छोटा कदम लें, और यही बार-बार दोहराएँ।

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$

यहाँ $\alpha$ **learning rate** है, यानी कदम का size। एक example के लिए derivative निकालें तो:

$$\frac{\partial}{\partial \theta_j} \frac{1}{2}\left(h_\theta(x) - y\right)^2 = \left(h_\theta(x) - y\right) x_j$$

इससे **LMS (Least Mean Squares) update rule** मिलता है, जिसे Widrow-Hoff rule भी कहते हैं:

$$\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}$$

ध्यान दें कि update error $(y - h_\theta(x))$ के अनुपात में है: prediction लगभग सही हो तो बदलाव छोटा, बहुत गलत हो तो बदलाव बड़ा।

### Batch बनाम Stochastic

- **Batch Gradient Descent:** हर step में पूरे training set का जोड़ लेते हैं: $\theta_j := \theta_j + \alpha \sum_{i=1}^{m} (y^{(i)} - h_\theta(x^{(i)}))\, x_j^{(i)}$। रास्ता smooth होता है, पर $m$ बड़ा हो तो हर step महँगा पड़ता है।
- **Stochastic Gradient Descent (SGD):** हर step में सिर्फ़ एक example देखकर update कर देते हैं। रास्ता टेढ़ा-मेढ़ा (noisy) होता है और minimum के आस-पास घूमता रहता है, पर बड़े data पर यह बहुत जल्दी अच्छे $\theta$ के पास पहुँच जाता है।

### Learning rate का असर

- $\alpha$ बहुत छोटा: सही दिशा, पर बहुत धीमा।
- $\alpha$ ठीक-ठाक: तेज़ी से minimum तक।
- $\alpha$ बहुत बड़ा: कदम minimum के पार निकल जाते हैं, $J$ घटने की जगह उछलता है या बढ़ता ही जाता है (diverge)।

## 5. Normal Equations

Linear Regression में minimum सीधे formula से भी निकल सकता है। सारे examples को matrix $X$ (हर row एक $x^{(i)T}$) और vector $\vec{y}$ में रखें। तब

$$J(\theta) = \frac{1}{2} (X\theta - \vec{y})^T (X\theta - \vec{y}), \qquad \nabla_\theta J(\theta) = X^T X \theta - X^T \vec{y}$$

Gradient को zero रखने पर **Normal Equations** मिलती हैं:

$$X^T X \theta = X^T \vec{y} \quad\Longrightarrow\quad \theta = (X^T X)^{-1} X^T \vec{y}$$

**Geometric मतलब:** $X\theta$ हमेशा $X$ के columns के span (एक plane) में रहता है। सबसे अच्छा $\theta$ वह है जिसके लिए $X\theta$, $\vec{y}$ का उस plane पर **projection** हो। तब residual $\vec{y} - X\theta$ plane के perpendicular होता है, यानी $X^T(\vec{y} - X\theta) = 0$। यही Normal Equation है।

| | Gradient Descent | Normal Equations |
|---|---|---|
| $\alpha$ चुनना | ज़रूरी | नहीं |
| Iterations | बहुत | एक ही बार में |
| $n$ बहुत बड़ा हो | ठीक चलता है | $(X^TX)^{-1}$ लगभग $O(n^3)$, धीमा |
| दूसरे models | लगभग हर जगह चलता है | सिर्फ़ Linear Regression जैसे cases |

## 6. Probabilistic Interpretation: squared error ही क्यों?

मान लीजिए $y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$, जहाँ noise $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$ है और सारे examples i.i.d. हैं। तब

$$p(y^{(i)} \mid x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\left( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right)$$

पूरे data का log-likelihood:

$$\ell(\theta) = \sum_{i=1}^{m} \log p(y^{(i)} \mid x^{(i)}; \theta) = m \log \frac{1}{\sqrt{2\pi}\,\sigma} - \frac{1}{\sigma^2} \cdot \frac{1}{2} \sum_{i=1}^{m} \left(y^{(i)} - \theta^T x^{(i)}\right)^2$$

$\ell(\theta)$ को maximize करना ठीक वही है जो $J(\theta)$ को minimize करना। यानी **Gaussian noise की assumption के साथ, Least Squares ही Maximum Likelihood Estimate है**। जवाब $\sigma^2$ पर निर्भर नहीं करता।

## 7. Practical बातें और आम गलतियाँ

- **Feature scaling:** अगर एक feature हज़ारों में है और दूसरा इकाइयों में, तो $J$ का कटोरा बहुत लंबा-पतला हो जाता है और Gradient Descent zig-zag करता है। हर feature को mean 0 और standard deviation 1 पर ले आएँ।
- **$J$ को हर iteration पर plot करें:** सही $\alpha$ पर $J$ हर step में घटना चाहिए। बढ़ रहा है तो $\alpha$ छोटा करें।
- **$X^TX$ invertible न हो** (जैसे duplicate features या $m < n$): pseudo-inverse या regularization लगाएँ (Lecture 8)।
- **Linear का मतलब parameters में linear:** $x^2$, $\log x$ जैसे नए features जोड़कर यही model curves भी fit कर सकता है।

## 8. खुद से पूछें

1. $J(\theta)$ में $\frac{1}{2}$ क्यों रखा गया है? (संकेत: derivative लेते समय क्या होता है?)
2. अगर सारे examples के लिए $h_\theta(x^{(i)}) = y^{(i)}$ हो जाए, तो LMS update क्या करेगा?
3. SGD minimum पर पूरी तरह रुकता क्यों नहीं, और इसे रोकने के लिए $\alpha$ के साथ क्या किया जा सकता है?
