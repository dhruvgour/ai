# Introduction और Basic Concepts

> **इस lecture में:** Machine Learning क्या है (Arthur Samuel और Tom Mitchell की definitions), इसके तीन बड़े परिवार (Supervised Learning, Unsupervised Learning, Reinforcement Learning), वह शब्दावली जो पूरी किताब में चलेगी (feature, label, training set, hypothesis, parameters, cost function, generalization, overfitting, train/test split), एक ML project का आम workflow, और आगे के lectures का roadmap।

## 1. एक line में idea

आम programming में हम computer को नियम खुद लिखकर देते हैं: "अगर email में ये शब्द हों, तो उसे spam मानो।" Machine Learning में हम नियम नहीं लिखते। हम computer को बहुत सारे उदाहरण (data) देते हैं, और एक ऐसा algorithm देते हैं जो उन उदाहरणों से नियम खुद निकाल लेता है।

| | हम क्या देते हैं | computer क्या लौटाता है |
|---|---|---|
| आम programming | data + नियम (program) | जवाब |
| Machine Learning | data + सही जवाब (उदाहरण) | नियम (model) |

यह तरीका तब काम आता है जब नियम हाथ से लिखना बहुत मुश्किल हो (चेहरा पहचानना, आवाज़ समझना, गाड़ी चलाना), पर उदाहरण इकट्ठा करना आसान हो।

## 2. Machine Learning की दो definitions

**Arthur Samuel (1959):** Machine Learning वह field of study है जो computers को यह क्षमता देता है कि वे बिना explicitly program किए सीख सकें।

Samuel ने checkers खेलने वाला program बनाया था। Program ने खुद के खिलाफ़ हज़ारों games खेले, और धीरे-धीरे पहचानना सीख लिया कि board की कौन सी स्थितियाँ जीत की ओर ले जाती हैं। अंत में वह Samuel से बेहतर खेलने लगा, जबकि Samuel ने उसे कोई चाल नहीं सिखाई थी।

**Tom Mitchell (1997):** एक computer program किसी task $T$ और performance measure $P$ के लिए experience $E$ से **सीखता है**, अगर $T$ पर उसका performance, $P$ से नापने पर, $E$ के साथ बेहतर होता जाए।

यह definition ज़्यादा काम की है, क्योंकि यह हमें तीन सवाल पूछने पर मजबूर करती है: काम क्या है, अच्छा या बुरा कैसे नापेंगे, और सीखने के लिए data क्या है।

| Problem | Task $T$ | Performance $P$ | Experience $E$ |
|---|---|---|---|
| Checkers | checkers खेलना | जीते हुए games का हिस्सा | खुद के खिलाफ़ खेले गए games |
| Spam filter | हर email को spam / not spam बताना | नए emails में से कितने % सही पहचाने गए | वे emails जिन पर user ने खुद spam / not spam का निशान लगाया |
| घर की कीमत | घर की कीमत का अंदाज़ा लगाना | नए घरों पर औसत squared error (जितना कम, उतना अच्छा) | पहले बिक चुके घरों का data |

**Worked example (spam filter):** मान लीजिए 100 labelled emails देखने के बाद program नए emails में से 80% सही पहचानता है, और 10,000 emails देखने के बाद 97%। $E$ बढ़ा, तो $T$ पर $P$ बेहतर हुआ। Mitchell की definition के हिसाब से program ने सीखा है। अगर accuracy 80% पर ही अटकी रहती, तो program सिर्फ़ data जमा कर रहा होता, सीख नहीं रहा होता। ध्यान दें कि $P$ हमेशा **नए** emails पर नापा जाता है, उन पर नहीं जिन्हें program पहले देख चुका है।

## 3. शब्दावली और Notation

पूरी किताब में एक ही उदाहरण याद रखें: घर के size और bedrooms से उसकी कीमत का अंदाज़ा लगाना।

| शब्द | Symbol | मतलब | घर वाले उदाहरण में |
|---|---|---|---|
| feature | $x_j$ | input का एक नापा हुआ गुण | size, bedrooms की संख्या |
| input (feature vector) | $x \in \mathcal{X}$ | एक उदाहरण के सारे features | $x = (2000, 3)$ |
| label / target | $y \in \mathcal{Y}$ | वह सही जवाब जो हमें predict करना है | कीमत |
| training example | $(x^{(i)}, y^{(i)})$ | input और उसके सही जवाब का एक जोड़ा | एक बिका हुआ घर |
| training set | $\{(x^{(i)}, y^{(i)});\ i = 1, \dots, m\}$ | सीखने के लिए दिए गए सारे examples | $m$ घरों की list |
| hypothesis / model | $h: \mathcal{X} \to \mathcal{Y}$ | वह function जो $x$ से $y$ का अंदाज़ा देता है | कीमत का formula |
| parameters | $\theta$ | model के अंदर की वे संख्याएँ जो data से सीखी जाती हैं | formula के weights |
| cost / loss function | $J(\theta)$ | एक संख्या जो बताती है कि model कितना गलत है | औसत squared error |

यहाँ $m$ training examples की संख्या है और $n$ features की संख्या। $x^{(i)}$ में ऊपर लिखा $(i)$ सिर्फ़ example का नंबर है, power नहीं। $\mathcal{X}$ सारे संभव inputs का set है और $\mathcal{Y}$ सारे संभव outputs का।

## 4. Supervised Learning

Supervised Learning में हर training example के साथ सही जवाब $y^{(i)}$ दिया होता है, जैसे कोई शिक्षक (supervisor) हर सवाल का हल साथ में दे रहा हो। लक्ष्य है training set से ऐसा $h$ सीखना कि $h(x)$, $y$ का अच्छा अंदाज़ा हो। $y$ किस तरह का है, इससे दो तरह की problems बनती हैं:

| | Regression | Classification |
|---|---|---|
| $y$ कैसा है | continuous संख्या, $y \in \mathbb{R}$ | गिनी-चुनी categories में से एक, जैसे $y \in \{0, 1\}$ |
| उदाहरण | घर की कीमत, कल का तापमान | spam / not spam, tumor malignant / benign |
| model क्या सीखता है | data के पास से गुज़रने वाली line / surface | classes को अलग करने वाली **decision boundary** |
| आम loss | squared error $(h(x) - y)^2$ | गलत हुआ या नहीं: $1\{h(x) \neq y\}$ |

यहाँ $1\{\cdot\}$ indicator function है: अंदर की बात सच हो तो 1, वरना 0। Classification में दो classes हों तो उसे binary classification कहते हैं, ज़्यादा हों तो multi-class।

## 5. Cost Function, Training और Prediction

Model कितना गलत है, यह एक संख्या में नापने के लिए हम हर example पर loss $L$ निकालते हैं और उसका औसत लेते हैं:

$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} L\left( h_\theta(x^{(i)}),\, y^{(i)} \right)$$

- **Training (learning):** वह $\theta$ ढूँढना जो cost को सबसे छोटा कर दे: $\hat{\theta} = \arg\min_\theta J(\theta)$। यह काम एक बार होता है और इसमें पूरा training set लगता है।
- **Prediction (inference):** सीखे हुए $\hat{\theta}$ को स्थिर रखकर किसी नए $x$ के लिए $\hat{y} = h_{\hat{\theta}}(x)$ निकालना। इसमें training set की ज़रूरत नहीं पड़ती।

अलग-अलग lectures में $J$ के आगे का constant अलग दिखेगा (जैसे Lecture 2 में $\frac{1}{2}$)। इससे minimum की जगह नहीं बदलती, सिर्फ़ maths साफ़ होता है।

## 6. Generalization, Underfitting और Overfitting

हमारा असली लक्ष्य training set को याद करना नहीं है। लक्ष्य है कि model उन examples पर अच्छा चले जिन्हें उसने कभी नहीं देखा। इसी क्षमता को **generalization** कहते हैं। मान लें सारे examples किसी अनजान distribution $\mathcal{D}$ से आते हैं। तब दो अलग errors हैं:

$$\hat{\varepsilon}_{\text{train}}(h) = \frac{1}{m} \sum_{i=1}^{m} L\left( h(x^{(i)}), y^{(i)} \right), \qquad \varepsilon(h) = \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ L\left( h(x), y \right) \right]$$

पहला **training error** है, जिसे हम नाप सकते हैं। दूसरा **generalization error** है, जो हमें चाहिए, पर जिसे सीधे नहीं नापा जा सकता क्योंकि $\mathcal{D}$ हमें पता नहीं। इसका अंदाज़ा लगाने के लिए **train/test split** करते हैं: data को random तरीके से दो हिस्सों में बाँटें (जैसे 70% और 30%), बड़े हिस्से पर train करें, और छोटे हिस्से (**test set**) को training से पूरी तरह अलग रखकर सिर्फ़ अंत में उस पर error नापें। Test set पर नापा गया error, $\varepsilon(h)$ का ईमानदार अंदाज़ा है।

| स्थिति | Training error | Test error | कारण | इलाज की दिशा |
|---|---|---|---|---|
| **Underfitting** | बड़ा | बड़ा | model बहुत सरल है, data का pattern पकड़ ही नहीं पाता | ज़्यादा features, ज़्यादा लचीला model |
| ठीक fit | छोटा | छोटा, training error के पास | model की complexity data के लायक है | कुछ नहीं |
| **Overfitting** | बहुत छोटा | बड़ा | model इतना लचीला है कि training set का noise भी याद कर लेता है | ज़्यादा data, सरल model, regularization |

Training error और test error के बीच का फ़ासला ही overfitting की पहचान है। Lecture 8 में यही बात bias और variance की भाषा में दोबारा आएगी।

## 7. Unsupervised Learning

यहाँ data में सिर्फ़ inputs हैं, कोई label नहीं: $\{x^{(1)}, \dots, x^{(m)}\}$। कोई "सही जवाब" नहीं है। लक्ष्य data के अंदर छिपा ढाँचा खोजना है। दो मुख्य काम:

- **Clustering:** मिलते-जुलते examples को झुंडों में बाँटना। जैसे ग्राहकों को उनकी खरीदारी की आदतों से groups में बाँटना। k-means algorithm हर example $x^{(i)}$ को किसी cluster $c^{(i)}$ में रखता है और हर cluster का केंद्र (centroid) $\mu_j$ चुनता है, ताकि यह दूरी छोटी से छोटी हो: $J(c, \mu) = \sum_{i=1}^{m} \| x^{(i)} - \mu_{c^{(i)}} \|^2$।
- **Dimensionality reduction:** $x \in \mathbb{R}^n$ को एक छोटे vector $z \in \mathbb{R}^k$ ($k < n$) में दबाना, ताकि ज़रूरी जानकारी (ज़्यादातर variance) बची रहे। यह data को 2D या 3D में देखने, noise हटाने और आगे के algorithms को तेज़ करने में काम आता है। PCA इसका सबसे जाना-माना तरीका है।

## 8. Reinforcement Learning

यहाँ कोई training set पहले से नहीं होता। एक **agent** किसी environment में कदम-दर-कदम फ़ैसले लेता है। समय $t$ पर वह state $s_t$ में है, action $a_t$ चुनता है, environment उसे नई state $s_{t+1}$ में पहुँचा देता है, और agent को एक संख्या मिलती है: **reward** $R(s_t)$। Agent को कोई नहीं बताता कि सही action क्या था। उसे बस reward दिखता है, और वह भी अक्सर देर से (शतरंज में जीत या हार का पता आखिरी चाल पर चलता है)। लक्ष्य एक **policy** $\pi$ सीखना है, यानी state से action का नियम, जो लंबे समय का कुल discounted reward बड़ा से बड़ा करे:

$$\mathbb{E}\left[ R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \dots \right], \qquad 0 \le \gamma < 1$$

$\gamma$ **discount factor** है: जल्दी मिलने वाला reward, देर से मिलने वाले reward से ज़्यादा कीमती गिना जाता है। उदाहरण: robot का चलना सीखना, games खेलना, helicopter उड़ाना।

| | Supervised | Unsupervised | Reinforcement |
|---|---|---|---|
| Data | $(x^{(i)}, y^{(i)})$ के जोड़े | सिर्फ़ $x^{(i)}$ | agent के अपने अनुभव: states, actions, rewards |
| सीखने का संकेत | हर example का सही जवाब | कोई नहीं | reward, अक्सर देर से |
| लक्ष्य | नए $x$ के लिए $y$ predict करना | data का ढाँचा खोजना | अच्छे फ़ैसलों का सिलसिला (policy) सीखना |

## 9. एक ML project का workflow

1. **Problem तय करें:** $T$ क्या है, $P$ क्या है? यह regression है, classification है, या कुछ और?
2. **Data इकट्ठा करें:** यही आपका $E$ है। Data वैसा ही होना चाहिए जैसा model को असल इस्तेमाल में मिलेगा।
3. **Data देखें और features बनाएँ:** plots बनाएँ, गलत या गायब values ठीक करें, features को scale करें।
4. **Split करें:** training set और test set अलग कर लें, और test set को ताले में रख दें।
5. **सरल model से शुरू करें:** पहले एक baseline (जैसे Linear या Logistic Regression) train करें।
6. **Train करें:** training set पर $J(\theta)$ minimize करें।
7. **जाँचें और सुधारें:** training error और held-out error की तुलना करें। Underfitting है या overfitting? उसी हिसाब से model, features या data बदलें और दोहराएँ (Lecture 12 इसी पर है)।
8. **अंतिम परीक्षा और इस्तेमाल:** सब तय हो जाने के बाद एक बार test set पर error नापें, फिर model को काम पर लगाएँ और उस पर नज़र रखें।

## 10. Roadmap: आगे क्या आएगा

| Lectures | विषय | परिवार |
|---|---|---|
| 2 से 4 | Linear models और GLMs: Linear Regression, Logistic Regression, Newton's Method, Perceptron, Exponential Family, Softmax Regression | Supervised |
| 5 से 6 | Generative models (जैसे Gaussian Discriminant Analysis, Naive Bayes) | Supervised |
| 6 से 7 | Support Vector Machines (SVM) और kernels | Supervised |
| 8 | Bias-variance tradeoff और regularization | सीखने का theory |
| 9 | Decision Trees और ensembles | Supervised |
| 10 से 11 | Neural Networks | Supervised |
| 12 | Practical advice: ML project को debug करना और सुधारना | सभी |
| 13 से 15 | Unsupervised Learning (जैसे k-means, EM algorithm, PCA) | Unsupervised |
| 16 से 19 | Reinforcement Learning (जैसे MDPs, value iteration) | Reinforcement |
| 20 | Wrap-up और Convolutional Neural Networks (CNNs) | Supervised |

## 11. Practical बातें और आम गलतियाँ

- **Test set पर train न करें, और उसे देखकर फ़ैसले भी न लें:** अगर आप test error देख-देखकर model बदलते रहे, तो test set भी चुपचाप training का हिस्सा बन जाता है और उसका error ज़रूरत से ज़्यादा अच्छा दिखने लगता है। Model चुनने के लिए एक अलग validation set रखें (Lecture 8)।
- **Split से पहले shuffle करें:** अगर data किसी क्रम में है (जैसे तारीख या class के हिसाब से), तो बिना shuffle किए काटने पर train और test अलग-अलग तरह के हो जाएँगे।
- **Training error कम होना काफ़ी नहीं:** 100% training accuracy अक्सर overfitting का संकेत है, सफलता का नहीं। हमेशा held-out data पर नापें।
- **Accuracy धोखा दे सकती है:** अगर 99% emails spam नहीं हैं, तो "कभी spam नहीं" कहने वाला बेकार model भी 99% accurate है। $P$ सोच-समझकर चुनें।
- **पहले सरल model:** एक सरल baseline जल्दी बता देता है कि problem कितनी कठिन है और data में कोई गड़बड़ तो नहीं।
- **Data की quality, algorithm से ज़्यादा मायने रखती है:** गलत labels या असल दुनिया से अलग data पर कोई भी algorithm अच्छा नहीं चलेगा।

## 12. खुद से पूछें

1. एक program मौसम के पुराने records देखकर कल की बारिश की probability बताता है। Mitchell की definition में यहाँ $T$, $P$ और $E$ क्या हैं? यह regression है या classification?
2. किसी model का training error 1% है और test error 25%। यह underfitting है या overfitting? आप कौन सी दो चीज़ें आज़माएँगे?
3. Supervised Learning और Reinforcement Learning दोनों में सीखने का कोई संकेत मिलता है। फिर reward, label से किस तरह अलग है?
