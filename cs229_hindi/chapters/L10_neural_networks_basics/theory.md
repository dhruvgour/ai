# Neural Networks: Basics

> **इस lecture में:** Logistic Regression से एक neuron तक का रास्ता, hidden units और end-to-end learning, layers का notation और forward propagation, activation functions (sigmoid, tanh, ReLU) और non-linearity क्यों ज़रूरी है, vectorization, output layers और उनके loss functions, parameters गिनना, XOR problem, और universal approximation का idea।

## 1. एक line में idea

Logistic Regression एक अकेली इकाई है: वह inputs का weighted sum लेती है और उसे sigmoid से गुज़ार देती है। ऐसी इकाई को **neuron** कहते हैं। अकेला neuron सिर्फ़ सीधी decision boundary (line या plane) बना सकता है। **Neural network** में हम बहुत सारे neurons को layers में जोड़ देते हैं: एक layer के outputs अगली layer के inputs बन जाते हैं। बीच की layers data से अपने काम के features खुद सीख लेती हैं, इसलिए पूरा network टेढ़ी-मेढ़ी boundaries और जटिल functions भी सीख सकता है।

इस lecture में हम सिर्फ़ यह देखेंगे कि network prediction कैसे निकालता है (**forward propagation**)। Parameters कैसे सीखे जाते हैं (backpropagation), यह अगले lecture का विषय है।

## 2. Logistic Regression से एक neuron तक

Logistic Regression में $h_\theta(x) = g(\theta^T x)$ था। Neural networks में intercept को अलग लिखने का रिवाज है: weights $w$ और bias $b$। एक neuron दो काम करता है:

$$z = w^T x + b = \sum_{j=1}^{n} w_j x_j + b, \qquad a = g(z)$$

- $z$ को **pre-activation** कहते हैं: यह inputs का weighted sum है।
- $g$ **activation function** है, और $a$ neuron का output है, जिसे **activation** कहते हैं।

$g$ अगर sigmoid हो, तो यह neuron ठीक Logistic Regression है। $w$ की दिशा बताती है कि output किस तरफ़ बढ़ता है, $w$ का size बताता है कि बदलाव कितना तीखा है, और $b$ boundary को आगे-पीछे खिसकाता है।

## 3. Hidden units: घर की कीमत वाला example

मान लीजिए घर की कीमत का अंदाज़ा लगाना है और inputs चार हैं: size, bedrooms की संख्या, zip code, और मोहल्ले की wealth। एक समझदार इंसान पहले बीच के कुछ features सोचेगा:

- size और bedrooms से पता चलता है कि घर कितने बड़े **family** के लिए ठीक है।
- zip code से पता चलता है कि इलाका कितना **walkable** है।
- zip code और wealth से **school quality** का अंदाज़ा लगता है।

कीमत इन तीन बीच वाले features पर निर्भर करती है। हर बीच वाले feature को एक neuron मान लें, तो यह एक छोटा neural network है: 4 inputs, 3 **hidden units**, 1 output।

असली neural network में हम यह नहीं बताते कि कौन सा hidden unit क्या नापेगा, और न यह कि वह किन inputs को देखेगा। हर hidden unit **सारे** inputs से जुड़ा होता है (fully-connected layer)। हम सिर्फ़ $(x, y)$ के जोड़े देते हैं, और training खुद तय करती है कि बीच के features क्या हों। इसे **end-to-end learning** कहते हैं। "Hidden" नाम इसीलिए है: training set में इन units का सही मान कहीं नहीं लिखा होता।

## 4. Layers, Notation और Forward Propagation

Layer का number square bracket में ऊपर लिखा जाता है, ताकि वह example के number $x^{(i)}$ से अलग दिखे।

| Symbol | मतलब |
|---|---|
| $L$ | layers की संख्या (input layer को गिना नहीं जाता) |
| $n^{[l]}$ | layer $l$ में units की संख्या, और $n^{[0]} = n$ |
| $W^{[l]}$ | layer $l$ का weight matrix, size $n^{[l]} \times n^{[l-1]}$ |
| $b^{[l]}$ | layer $l$ का bias vector, size $n^{[l]}$ |
| $z^{[l]}$ | layer $l$ के pre-activations (vector) |
| $a^{[l]}$ | layer $l$ के activations, और $a^{[0]} = x$ |
| $g^{[l]}$ | layer $l$ का activation function |

$W^{[l]}$ की $j$-वीं row, layer $l$ के $j$-वें neuron का weight vector है। यानी $W^{[l]}_{jk}$ वह weight है जो पिछली layer के unit $k$ को इस layer के unit $j$ से जोड़ता है।

**2-layer network** (एक hidden layer और एक output layer) का forward propagation:

$$\begin{aligned} z^{[1]} &= W^{[1]} x + b^{[1]}, & a^{[1]} &= g^{[1]}(z^{[1]}) \\ z^{[2]} &= W^{[2]} a^{[1]} + b^{[2]}, & a^{[2]} &= g^{[2]}(z^{[2]}) = h_\theta(x) \end{aligned}$$

यहाँ $g$ vector के हर element पर अलग-अलग लगता है। **General रूप** में, $l = 1, 2, \dots, L$ के लिए बारी-बारी से:

$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}, \qquad a^{[l]} = g^{[l]}(z^{[l]})$$

आख़िरी activation $a^{[L]}$ ही network का prediction है। हर layer बस वही करती है जो एक neuron करता है, फ़र्क इतना है कि कई neurons एक साथ matrix के रूप में लिखे गए हैं।

## 5. Activation functions

| नाम | $g(z)$ | Range | $g'(z)$ |
|---|---|---|---|
| sigmoid | $\dfrac{1}{1 + e^{-z}}$ | $(0, 1)$ | $g(z)\,(1 - g(z))$ |
| tanh | $\dfrac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$ | $(-1, 1)$ | $1 - g(z)^2$ |
| ReLU | $\max(0, z)$ | $[0, \infty)$ | $z > 0$ पर $1$, $z < 0$ पर $0$ |

- **sigmoid** का output probability की तरह पढ़ा जा सकता है, इसलिए binary classification की output layer में यही लगता है। Hidden layers में यह कम पसंद किया जाता है, क्योंकि बड़े $\lvert z \rvert$ पर यह सपाट हो जाता है (saturate) और derivative लगभग zero हो जाता है।
- **tanh** असल में खिंचा हुआ sigmoid है: $\tanh(z) = 2\,\sigma(2z) - 1$, जहाँ $\sigma$ sigmoid है। इसका output zero के दोनों तरफ़ फैला होता है, जो अगली layer के लिए अच्छा है। पर saturation की दिक्कत इसमें भी है।
- **ReLU** (Rectified Linear Unit) सबसे सरल है: negative को zero कर दो, positive को वैसा ही रहने दो। Positive हिस्से में derivative हमेशा $1$ रहता है, इसलिए गहरे networks में gradients कमज़ोर नहीं पड़ते। आजकल hidden layers के लिए यही पहली पसंद है।

### Non-linearity क्यों ज़रूरी है

मान लीजिए हम activation हटा दें, यानी हर जगह $g(z) = z$ रखें। तब 2-layer network यह निकालता है:

$$\begin{aligned} a^{[2]} &= W^{[2]} a^{[1]} + b^{[2]} = W^{[2]} \left( W^{[1]} x + b^{[1]} \right) + b^{[2]} \\ &= \underbrace{W^{[2]} W^{[1]}}_{\tilde{W}}\, x + \underbrace{W^{[2]} b^{[1]} + b^{[2]}}_{\tilde{b}} = \tilde{W} x + \tilde{b} \end{aligned}$$

यह फिर से $x$ का एक linear function है। यही तर्क दोहराने पर 100 linear layers भी सिमटकर एक ही linear map बन जाती हैं: $\tilde{W} = W^{[L]} \cdots W^{[2]} W^{[1]}$। यानी non-linear activation के बिना गहराई का कोई फ़ायदा नहीं, और network Linear Regression (या sigmoid output के साथ Logistic Regression) से ज़्यादा कुछ नहीं कर सकता।

## 6. Vectorization: सारे examples एक साथ

$m$ examples पर Python का loop चलाना बहुत धीमा है। इसकी जगह सारे examples को एक matrix के columns बना लें:

$$X = \begin{bmatrix} x^{(1)} & x^{(2)} & \cdots & x^{(m)} \end{bmatrix} \in \mathbb{R}^{n \times m}$$

तब पूरी layer एक matrix multiplication में निकल जाती है ($A^{[0]} = X$):

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}, \qquad A^{[l]} = g^{[l]}(Z^{[l]})$$

$Z^{[l]}$ का size $n^{[l]} \times m$ है, और इसका $i$-वाँ column ठीक $z^{[l](i)}$ है, क्योंकि matrix product $W A$ का $i$-वाँ column $W$ गुणा $A$ का $i$-वाँ column होता है। $b^{[l]}$ एक ही column है, पर उसे हर column में जोड़ना है। NumPy यह अपने आप कर देता है, इसे **broadcasting** कहते हैं। गणित की भाषा में यह $b^{[l]} \mathbf{1}^T$ जोड़ना है, जहाँ $\mathbf{1}$ में $m$ ones हैं।

गिनती वही रहती है, पर काम Python के loop की जगह optimized linear algebra code (और GPU) करता है। फ़र्क अक्सर 10 से 100 गुना या उससे भी ज़्यादा होता है।

ध्यान दें: Lecture 2 के design matrix में हर example एक **row** था। यहाँ notation को equations के अनुकूल रखने के लिए हर example एक **column** है।

## 7. Output layer और Loss

Hidden layers features बनाती हैं। आख़िरी layer problem के हिसाब से चुनी जाती है, और हर output layer के साथ उसका अपना loss आता है। नीचे $a = a^{[L]}$ और $z = z^{[L]}$ है।

| Problem | Output activation | Loss $\mathcal{L}(a, y)$ |
|---|---|---|
| Binary classification, $y \in \{0, 1\}$ | sigmoid: $a = \dfrac{1}{1 + e^{-z}}$ | log loss: $-\left[ y \log a + (1 - y) \log (1 - a) \right]$ |
| $K$ classes, $y$ one-hot vector | softmax: $a_k = \dfrac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$ | cross-entropy: $-\sum_{k=1}^{K} y_k \log a_k$ |
| Regression, $y \in \mathbb{R}$ | linear: $a = z$ | squared error: $\frac{1}{2} (a - y)^2$ |

ये तीनों जोड़े Lecture 3 और 4 से आए हैं: हर loss किसी distribution (Bernoulli, Multinomial, Gaussian) का negative log-likelihood है। यानी network को train करना भी Maximum Likelihood ही है। पूरे training set का cost हर example के loss का औसत है:

$$J(W, b) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}\left( a^{[L](i)}, y^{(i)} \right)$$

$J$ को घटाने के लिए हमें हर $W^{[l]}$ और $b^{[l]}$ के लिए gradient चाहिए। उसे निकालने का तरीका (backpropagation) अगले lecture में है।

## 8. Parameters गिनना

Layer $l$ में $n^{[l]} \times n^{[l-1]}$ weights और $n^{[l]}$ biases होते हैं। इसलिए कुल parameters:

$$\sum_{l=1}^{L} \left( n^{[l]}\, n^{[l-1]} + n^{[l]} \right)$$

उदाहरण: 64 inputs, 32 hidden units और 10 outputs वाले network में $(32 \cdot 64 + 32) + (10 \cdot 32 + 10) = 2080 + 330 = 2410$ parameters हैं। ध्यान दें कि गिनती पड़ोसी layers के sizes के गुणा से बढ़ती है, इसलिए चौड़ी layers बहुत जल्दी महँगी हो जाती हैं।

## 9. XOR: जहाँ एक neuron हार जाता है

XOR में चार points हैं: $(0,0) \to 0$, $(0,1) \to 1$, $(1,0) \to 1$, $(1,1) \to 0$। क्या कोई linear classifier, जो $w_1 x_1 + w_2 x_2 + b > 0$ होने पर $1$ बोले, इन्हें सही कर सकता है? चारों points से चार शर्तें मिलती हैं:

$$b \le 0, \qquad w_2 + b > 0, \qquad w_1 + b > 0, \qquad w_1 + w_2 + b \le 0$$

बीच की दो शर्तें जोड़ने पर $w_1 + w_2 + 2b > 0$, यानी $w_1 + w_2 + b > -b \ge 0$। यह चौथी शर्त से टकराता है। इसलिए **कोई भी** line XOR को अलग नहीं कर सकती।

अब दो ReLU hidden units लें: $a^{[1]}_1 = \max(0,\, x_1 + x_2)$ और $a^{[1]}_2 = \max(0,\, x_1 + x_2 - 1)$, और output $a^{[1]}_1 - 2 a^{[1]}_2$ रखें।

| $x_1$ | $x_2$ | $y$ | $a^{[1]}_1$ | $a^{[1]}_2$ | $a^{[1]}_1 - 2 a^{[1]}_2$ |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 1 | 0 | 1 |
| 1 | 0 | 1 | 1 | 0 | 1 |
| 1 | 1 | 0 | 2 | 1 | 0 |

Output ठीक $y$ के बराबर है। असली बात यह है कि hidden layer ने points को एक नई जगह $(a^{[1]}_1, a^{[1]}_2)$ में पहुँचा दिया, जहाँ वे $(0,0)$, $(1,0)$, $(1,0)$ और $(2,1)$ हैं। इस नई जगह में एक सीधी line उन्हें अलग कर देती है। Hidden layer का काम यही है: data को ऐसे मोड़ना कि आख़िरी layer का linear classifier काम कर सके।

## 10. Universal Approximation

**Idea (informal):** एक hidden layer वाला network, जिसमें activation non-linear हो (sigmoid, tanh, ReLU) और hidden units काफ़ी हों, किसी भी continuous function को एक सीमित region पर जितनी चाहें उतनी accuracy से approximate कर सकता है।

ReLU के साथ इसे ऐसे सोचें: हर hidden unit surface में एक **मोड़** (fold) जोड़ता है, यानी एक line जिसके एक तरफ़ वह zero है और दूसरी तरफ़ ढलान है। बहुत सारे मोड़ जोड़ने पर टुकड़ों में सपाट (piecewise linear) surface बनती है, जो किसी भी चिकनी surface से चिपक सकती है। जितने ज़्यादा units, उतने छोटे टुकड़े, उतना अच्छा fit।

यह result तीन बातें **नहीं** कहता:

- कितने units चाहिए (संख्या बहुत बड़ी हो सकती है)।
- Gradient Descent वे weights ढूँढ ही लेगा।
- सीखा हुआ function नए data पर भी अच्छा चलेगा (overfitting का सवाल बना रहता है)।

Practice में कई layers वाला (deep) network अक्सर वही function बहुत कम units में बना लेता है, क्योंकि हर layer पिछली layer के features को दोबारा इस्तेमाल करती है।

## 11. Practical बातें और आम गलतियाँ

- **Shapes जाँचें:** सबसे आम bug गलत shape है। $W^{[l]}$ का size $n^{[l]} \times n^{[l-1]}$, $A^{[l]}$ का $n^{[l]} \times m$, और $b^{[l]}$ का $n^{[l]} \times 1$ होना चाहिए। Code में `assert` लगाने से घंटों बचते हैं।
- **Layers की गिनती:** "2-layer network" का मतलब एक hidden layer और एक output layer है। Input layer नहीं गिनी जाती।
- **Hidden layer में linear activation न रखें:** section 5 के हिसाब से वह layer बेकार हो जाती है।
- **Hidden layers में ReLU से शुरू करें।** sigmoid को binary output के लिए और softmax को multi-class output के लिए रखें।
- **Softmax को संभलकर निकालें:** बड़े $z$ पर $e^{z}$ overflow कर जाता है। exponent लेने से पहले $z$ में से उसका maximum घटा दें, जवाब नहीं बदलता। $\log 0$ से बचने के लिए $a$ को बहुत छोटे number से नीचे न जाने दें।
- **Weights को zero से शुरू न करें:** तब सारे hidden units एक जैसे रह जाते हैं। छोटे random numbers से शुरू करें (कारण अगले lecture में)।
- **Inputs को scale करें** (mean 0, standard deviation 1), जैसे Lecture 2 में किया था।

## 12. खुद से पूछें

1. एक network में 10 inputs, 5-5 units की दो hidden layers और 1 output है। इसमें कुल कितने parameters हैं? अगर सारे activations linear हों, तो वही function कितने parameters में लिखा जा सकता है?
2. Vectorized रूप में $Z^{[1]} = W^{[1]} X + b^{[1]}$ है। अगर $n = 3$, $n^{[1]} = 4$ और $m = 100$ हो, तो $W^{[1]}$, $X$, $b^{[1]}$ और $Z^{[1]}$ के shapes क्या हैं?
3. XOR वाले example में hidden units ReLU की जगह linear होते, तो $(a^{[1]}_1, a^{[1]}_2)$ वाली जगह में points कहाँ पहुँचते? क्या तब भी एक line उन्हें अलग कर पाती?
