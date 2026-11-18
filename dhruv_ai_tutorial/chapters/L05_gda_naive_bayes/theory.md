# Gaussian Discriminant Analysis और Naive Bayes

> **इस lecture में:** Discriminative और Generative learning का फ़र्क, Bayes rule से prediction, multivariate Gaussian, Gaussian Discriminant Analysis (GDA) और उसके Maximum Likelihood estimates, GDA की linear decision boundary, GDA बनाम Logistic Regression, और discrete features के लिए Naive Bayes (spam filter का उदाहरण)।

## 1. एक line में idea

अब तक के algorithms (जैसे Logistic Regression) सीधे यह सीखते थे कि $x$ दिया हो तो $y$ क्या होगा। वे दो classes के बीच की सीमा ढूँढते थे। **Generative** algorithm उल्टी तरफ़ से सोचता है: वह हर class का अलग नक्शा बनाता है कि उस class के examples कैसे दिखते हैं। नया $x$ आए तो वह पूछता है कि यह किस class के नक्शे से ज़्यादा मेल खाता है।

मान लीजिए हाथी और कुत्ते पहचानने हैं। Discriminative तरीका दोनों को अलग करने वाली रेखा ढूँढता है। Generative तरीका पहले हाथियों को देखकर हाथी का model बनाता है, फिर कुत्तों को देखकर कुत्ते का model, और नए जानवर को दोनों models से मिलाकर देखता है।

इस lecture में दो generative algorithms हैं। **GDA** तब काम आता है जब features continuous हों और हर class का data Gaussian जैसा हो। **Naive Bayes** तब, जब features discrete हों, जैसे "email में यह शब्द है या नहीं"।

## 2. Discriminative बनाम Generative

| | क्या सीखता है | उदाहरण |
|---|---|---|
| Discriminative | सीधे $p(y \mid x)$, या $x$ से label तक का map | Logistic Regression, Perceptron |
| Generative | $p(x \mid y)$ (हर class का model) और $p(y)$ (class prior) | GDA, Naive Bayes |

Generative model से prediction के लिए **Bayes rule** लगाते हैं:

$$p(y \mid x) = \frac{p(x \mid y)\, p(y)}{p(x)}, \qquad p(x) = p(x \mid y=1)\,p(y=1) + p(x \mid y=0)\,p(y=0)$$

अगर हमें सिर्फ़ सबसे संभावित class चाहिए, तो denominator $p(x)$ निकालने की ज़रूरत नहीं। वह $y$ पर निर्भर नहीं करता, इसलिए दोनों classes के लिए एक ही है और तुलना पर असर नहीं डालता:

$$\arg\max_y\, p(y \mid x) = \arg\max_y\, \frac{p(x \mid y)\, p(y)}{p(x)} = \arg\max_y\, p(x \mid y)\, p(y)$$

## 3. Multivariate Gaussian

$n$ dimensions का Gaussian $\mathcal{N}(\mu, \Sigma)$ दो चीज़ों से तय होता है: mean vector $\mu \in \mathbb{R}^n$ और covariance matrix $\Sigma \in \mathbb{R}^{n \times n}$ (symmetric और positive definite)। इसकी density है:

$$p(x; \mu, \Sigma) = \frac{1}{(2\pi)^{n/2}\, |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)$$

यहाँ $|\Sigma|$ का मतलब $\Sigma$ का determinant है। इस distribution के लिए $E[X] = \mu$ और $\mathrm{Cov}(X) = E\left[(X-\mu)(X-\mu)^T\right] = \Sigma$।

$\mu$ बदलने से पहाड़ी (density का उभार) सिर्फ़ खिसकती है, आकार नहीं बदलता। आकार $\Sigma$ तय करता है:

| $\Sigma$ | contours कैसे दिखते हैं |
|---|---|
| $I$ (identity) | गोल घेरे |
| $cI$, $c$ छोटा या बड़ा | गोल ही, पर छोटा $c$: पतली और ऊँची पहाड़ी; बड़ा $c$: चौड़ी और नीची (कुल volume हमेशा 1) |
| Diagonal, entries अलग-अलग | axes की दिशा में खिंचे ellipse |
| Off-diagonal entry positive | ellipse $x_1 = x_2$ की दिशा में झुका ($x_1$ बढ़े तो $x_2$ भी बढ़ता है) |
| Off-diagonal entry negative | ellipse उल्टी दिशा में झुका |

## 4. GDA Model

Classification problem है, $y \in \{0, 1\}$ और $x \in \mathbb{R}^n$। GDA यह मान लेता है:

$$\begin{aligned} y &\sim \mathrm{Bernoulli}(\phi) \\ x \mid y=0 &\sim \mathcal{N}(\mu_0, \Sigma) \\ x \mid y=1 &\sim \mathcal{N}(\mu_1, \Sigma) \end{aligned}$$

यानी $p(y) = \phi^y (1-\phi)^{1-y}$, और हर class एक Gaussian है। दोनों classes के mean अलग हैं ($\mu_0$, $\mu_1$), पर covariance $\Sigma$ **एक ही** है। Parameters हैं $\phi, \mu_0, \mu_1, \Sigma$।

### Joint log-likelihood

Logistic Regression में हम conditional likelihood $\prod_i p(y^{(i)} \mid x^{(i)})$ को maximize करते थे। Generative model में **joint** likelihood $\prod_i p(x^{(i)}, y^{(i)})$ को maximize करते हैं:

$$\begin{aligned} \ell(\phi, \mu_0, \mu_1, \Sigma) &= \log \prod_{i=1}^{m} p(x^{(i)}, y^{(i)}; \phi, \mu_0, \mu_1, \Sigma) \\ &= \sum_{i=1}^{m} \log p(x^{(i)} \mid y^{(i)}; \mu_0, \mu_1, \Sigma) + \sum_{i=1}^{m} \log p(y^{(i)}; \phi) \end{aligned}$$

### Maximum likelihood estimates

**$\phi$ के लिए:** सिर्फ़ दूसरा sum $\phi$ पर निर्भर है: $\sum_i \left[ y^{(i)} \log \phi + (1 - y^{(i)}) \log(1-\phi) \right]$। मान लें $m_1 = \sum_i 1\{y^{(i)}=1\}$। Derivative को zero रखें: $\frac{m_1}{\phi} - \frac{m - m_1}{1-\phi} = 0$, जिससे $\phi = m_1 / m$।

**$\mu_1$ के लिए:** सिर्फ़ $y^{(i)}=1$ वाले terms बचते हैं: $-\frac{1}{2} \sum_{i:\, y^{(i)}=1} (x^{(i)} - \mu_1)^T \Sigma^{-1} (x^{(i)} - \mu_1)$। इसका gradient $\Sigma^{-1} \sum_{i:\, y^{(i)}=1} (x^{(i)} - \mu_1)$ है। इसे zero रखने पर $\mu_1$ class 1 के examples का औसत निकलता है। $\mu_0$ के लिए भी यही।

**$\Sigma$ के लिए:** $\Sigma$ वाले terms हैं $-\frac{m}{2} \log|\Sigma| - \frac{1}{2} \sum_i (x^{(i)} - \mu_{y^{(i)}})^T \Sigma^{-1} (x^{(i)} - \mu_{y^{(i)}})$। अब $\nabla_A \log|A| = A^{-1}$ का इस्तेमाल करके $\Sigma^{-1}$ के सापेक्ष derivative लें और zero रखें: $\frac{m}{2}\Sigma - \frac{1}{2} \sum_i (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T = 0$।

सब मिलाकर:

$$\begin{aligned} \phi &= \frac{1}{m} \sum_{i=1}^{m} 1\{y^{(i)} = 1\} \\ \mu_0 &= \frac{\sum_{i=1}^{m} 1\{y^{(i)} = 0\}\, x^{(i)}}{\sum_{i=1}^{m} 1\{y^{(i)} = 0\}}, \qquad \mu_1 = \frac{\sum_{i=1}^{m} 1\{y^{(i)} = 1\}\, x^{(i)}}{\sum_{i=1}^{m} 1\{y^{(i)} = 1\}} \\ \Sigma &= \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T \end{aligned}$$

### Algorithm, step by step

1. गिनें कि कितने examples class 1 के हैं। यही अनुपात $\phi$ है।
2. हर class के examples का औसत निकालें: $\mu_0$ और $\mu_1$।
3. हर example में से उसकी अपनी class का mean घटाएँ, और इन centred vectors से एक साझा $\Sigma$ बनाएँ।
4. नए $x$ के लिए $p(x \mid y=1)\,\phi$ और $p(x \mid y=0)\,(1-\phi)$ निकालें। जो बड़ा हो, वही class।

कोई iteration नहीं, कोई learning rate नहीं: सब कुछ गिनती और औसत से मिल जाता है।

## 5. Decision boundary linear क्यों है?

Bayes rule से posterior को इस रूप में लिख सकते हैं:

$$p(y=1 \mid x) = \frac{p(x \mid y=1)\,\phi}{p(x \mid y=1)\,\phi + p(x \mid y=0)\,(1-\phi)} = \frac{1}{1 + \exp(-a(x))}, \qquad a(x) = \log \frac{p(x \mid y=1)\,\phi}{p(x \mid y=0)\,(1-\phi)}$$

दोनों Gaussians का normalizing constant एक ही है (क्योंकि $\Sigma$ एक ही है), इसलिए वह कट जाता है:

$$a(x) = -\frac{1}{2}(x-\mu_1)^T \Sigma^{-1} (x-\mu_1) + \frac{1}{2}(x-\mu_0)^T \Sigma^{-1} (x-\mu_0) + \log\frac{\phi}{1-\phi}$$

दोनों brackets खोलें। $-\frac{1}{2} x^T \Sigma^{-1} x$ और $+\frac{1}{2} x^T \Sigma^{-1} x$ आपस में कट जाते हैं। यही असली बात है: quadratic term गायब हो जाता है और बचता है

$$a(x) = \underbrace{(\mu_1 - \mu_0)^T \Sigma^{-1}}_{\theta^T}\, x + \underbrace{\frac{1}{2}\left( \mu_0^T \Sigma^{-1} \mu_0 - \mu_1^T \Sigma^{-1} \mu_1 \right) + \log\frac{\phi}{1-\phi}}_{\theta_0}$$

इसलिए

$$p(y=1 \mid x) = \frac{1}{1 + \exp\left(-(\theta^T x + \theta_0)\right)}, \qquad \theta = \Sigma^{-1}(\mu_1 - \mu_0)$$

Decision boundary वह जगह है जहाँ $p(y=1 \mid x) = 0.5$, यानी $\theta^T x + \theta_0 = 0$। यह एक सीधी line (या hyperplane) है। ध्यान दें कि prior $\phi$ सिर्फ़ $\theta_0$ में आता है: $\phi$ बढ़ाने से boundary की दिशा नहीं बदलती, वह बस class 0 की तरफ़ खिसक जाती है। अगर दोनों classes के $\Sigma$ अलग होते, तो quadratic term नहीं कटता और boundary curved होती।

## 6. GDA बनाम Logistic Regression

ऊपर हमने देखा कि GDA का posterior ठीक Logistic Regression वाले sigmoid रूप में है। पर दोनों algorithms एक ही data पर अलग-अलग $\theta$ देते हैं, क्योंकि वे अलग चीज़ maximize करते हैं।

- **GDA $\Rightarrow$ logistic posterior, पर उल्टा सच नहीं।** अगर $x \mid y$ Gaussian (साझा $\Sigma$) है तो $p(y \mid x)$ logistic है। पर $p(y \mid x)$ logistic होने से यह नहीं निकलता कि $x \mid y$ Gaussian है। जैसे, $x \mid y=0 \sim \mathrm{Poisson}(\lambda_0)$ और $x \mid y=1 \sim \mathrm{Poisson}(\lambda_1)$ हो, तब भी posterior logistic ही होता है।
- **GDA की assumptions ज़्यादा मज़बूत हैं।** जब वे सही (या लगभग सही) हों, तो GDA data का बेहतर इस्तेमाल करता है और कम training examples में अच्छा model दे देता है (asymptotically efficient)।
- **Logistic Regression ज़्यादा robust है।** वह कम मानकर चलता है, इसलिए data Gaussian न हो तब भी ठीक काम करता है। बड़े training set पर वह अक्सर GDA के बराबर या उससे बेहतर रहता है।

| | GDA | Logistic Regression |
|---|---|---|
| Maximize करता है | joint likelihood $\prod_i p(x^{(i)}, y^{(i)})$ | conditional likelihood $\prod_i p(y^{(i)} \mid x^{(i)})$ |
| Assumption | $x \mid y$ Gaussian, साझा $\Sigma$ | सिर्फ़ $p(y \mid x)$ का sigmoid रूप |
| Training | closed form (गिनती और औसत) | iterative (Gradient Ascent, Newton) |
| कम data, assumption सही | बेहतर | थोड़ा पीछे |
| Assumption गलत | बिगड़ सकता है | टिका रहता है |

## 7. Naive Bayes

अब features discrete हैं। उदाहरण: **spam filter**। एक vocabulary (शब्दों की सूची) लें जिसमें $n$ शब्द हैं। हर email को एक binary vector $x \in \{0,1\}^n$ से दिखाएँ: $x_j = 1$ अगर vocabulary का $j$-वाँ शब्द email में है, वरना $x_j = 0$। $y = 1$ का मतलब spam है।

अगर $n = 50000$ हो तो $x$ के $2^{50000}$ संभव values हैं। $p(x \mid y)$ को सीधे model करना असंभव है। इसलिए एक बहुत मज़बूत assumption लेते हैं।

**Naive Bayes assumption:** $y$ दिया हो, तो सारे $x_j$ आपस में conditionally independent हैं। यानी अगर पता है कि email spam है, तो उसमें "buy" होने की जानकारी से "price" होने की probability नहीं बदलती। Chain rule और इस assumption से:

$$\begin{aligned} p(x_1, \dots, x_n \mid y) &= p(x_1 \mid y)\, p(x_2 \mid y, x_1) \cdots p(x_n \mid y, x_1, \dots, x_{n-1}) \\ &= p(x_1 \mid y)\, p(x_2 \mid y) \cdots p(x_n \mid y) = \prod_{j=1}^{n} p(x_j \mid y) \end{aligned}$$

**Parameters:** $\phi_{j \mid y=1} = p(x_j = 1 \mid y = 1)$, $\phi_{j \mid y=0} = p(x_j = 1 \mid y = 0)$ और $\phi_y = p(y = 1)$। कुल $2n + 1$ parameters, $2^n$ की जगह।

Joint likelihood $\mathcal{L}(\phi_y, \phi_{j \mid y=0}, \phi_{j \mid y=1}) = \prod_{i=1}^{m} p(x^{(i)}, y^{(i)})$ है। इसका log लेने पर हर parameter का अपना अलग Bernoulli term बनता है, इसलिए हल ठीक GDA के $\phi$ जैसा निकलता है: हर estimate एक अनुपात है।

$$\begin{aligned} \phi_{j \mid y=1} &= \frac{\sum_{i=1}^{m} 1\{x_j^{(i)} = 1 \wedge y^{(i)} = 1\}}{\sum_{i=1}^{m} 1\{y^{(i)} = 1\}} \\ \phi_{j \mid y=0} &= \frac{\sum_{i=1}^{m} 1\{x_j^{(i)} = 1 \wedge y^{(i)} = 0\}}{\sum_{i=1}^{m} 1\{y^{(i)} = 0\}} \\ \phi_y &= \frac{1}{m} \sum_{i=1}^{m} 1\{y^{(i)} = 1\} \end{aligned}$$

शब्दों में: $\phi_{j \mid y=1}$ उन spam emails का हिस्सा है जिनमें शब्द $j$ आया है।

**Prediction:** नए email $x$ के लिए

$$p(y=1 \mid x) = \frac{\left( \prod_{j=1}^{n} p(x_j \mid y=1) \right) p(y=1)}{\left( \prod_{j=1}^{n} p(x_j \mid y=1) \right) p(y=1) + \left( \prod_{j=1}^{n} p(x_j \mid y=0) \right) p(y=0)}$$

जहाँ $p(x_j \mid y=1) = \phi_{j \mid y=1}$ अगर $x_j = 1$, और $1 - \phi_{j \mid y=1}$ अगर $x_j = 0$। जिस class का posterior बड़ा हो, वही चुनें।

### एक कमी: शून्य probability

मान लीजिए नए email में ऐसा शब्द है जो training set में कभी आया ही नहीं (vocabulary का शब्द $j$)। तब $\phi_{j \mid y=1} = 0$ और $\phi_{j \mid y=0} = 0$। दोनों products zero हो जाते हैं और posterior $\frac{0}{0}$ बन जाता है। सिर्फ़ इसलिए कि कोई घटना अब तक नहीं दिखी, उसकी probability zero मान लेना गलत है। अगला lecture इसे **Laplace smoothing** से ठीक करता है।

### Multinomial event model (झलक)

ऊपर वाला model (Bernoulli event model) सिर्फ़ यह देखता है कि शब्द आया या नहीं, कितनी बार आया यह नहीं। Text के लिए एक बेहतर model **multinomial event model** है। उसमें $x_j$ email के $j$-वें स्थान पर आए शब्द की पहचान है, $x_j \in \{1, \dots, |V|\}$, और $p(x, y) = p(y) \prod_{j} p(x_j \mid y)$। इस तरह शब्दों की गिनती भी काम आती है। इसका पूरा विवरण अगले lecture में है।

## 8. Practical बातें और आम गलतियाँ

- **Products की जगह logs जोड़ें:** हज़ारों छोटी probabilities का product computer में underflow होकर 0 बन जाता है। $\log p(y) + \sum_j \log p(x_j \mid y)$ की तुलना करें।
- **Gaussian assumption जाँचें:** हर class के features का histogram या scatter plot देखें। Data बहुत skewed हो या उसमें outliers हों, तो mean और $\Sigma$ बिगड़ते हैं और GDA की boundary भी। ऐसे में Logistic Regression सुरक्षित है।
- **$\Sigma$ साझा है:** $\Sigma$ बनाते समय हर example में से उसकी अपनी class का mean घटाएँ, पूरे data का mean नहीं।
- **$m$ छोटा, $n$ बड़ा:** तब $\Sigma$ singular हो सकता है और $\Sigma^{-1}$ नहीं निकलेगा।
- **Naive Bayes की probabilities पर ज़्यादा भरोसा न करें:** independence की assumption असल में गलत होती है, इसलिए posterior अक्सर 0 या 1 के बहुत पास आता है। फिर भी class का फ़ैसला अक्सर सही होता है, और algorithm बहुत तेज़ और सरल है।
- **Continuous feature पर Naive Bayes:** feature को कुछ buckets में बाँटकर discrete बना लें।

## 9. खुद से पूछें

1. $\arg\max_y p(y \mid x)$ निकालते समय $p(x)$ क्यों छोड़ सकते हैं? अगर आपको class नहीं, बल्कि probability की value चाहिए, तब भी छोड़ सकते हैं क्या?
2. GDA में दोनों classes का $\Sigma$ अलग-अलग ($\Sigma_0 \ne \Sigma_1$) हो, तो $a(x)$ में कौन सा term नहीं कटेगा और boundary कैसी दिखेगी?
3. Vocabulary में $n = 1000$ शब्द हैं। Naive Bayes assumption के बिना $p(x \mid y)$ के लिए कितने parameters चाहिए, और उसके साथ कितने?
