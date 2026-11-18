## यह guide कैसे पढ़ें

यह guide Stanford CS229 (Machine Learning) के syllabus के क्रम में चलती है: Lecture 1 से Lecture 20 तक। हर lecture का एक chapter है, और हर chapter के तीन भाग हैं:

| भाग | क्या है | कैसे इस्तेमाल करें |
|---|---|---|
| **भाग 1: Theory** | Idea, notation, maths और algorithm, हिंदी में | पहले इसे धीरे-धीरे पढ़ें। Equations को कागज़ पर खुद लिखकर देखें। |
| **भाग 2: 3D Animations** | हर chapter में कम से कम 5 animations | PDF में हर animation के तीन still frames हैं (शुरुआत, बीच में, अंत)। चलती animation देखने के लिए नीचे बताई files खोलें। |
| **भाग 3: Python Example** | NumPy से शुरू से लिखा एक चलने वाला example, उसके output के साथ | Code को खुद चलाएँ, फिर numbers बदलकर देखें कि क्या बदलता है। |

**भाषा के बारे में:** समझाने की भाषा हिंदी है, पर AI और Maths के शब्द (जैसे Gradient Descent, likelihood, kernel, variance) अंग्रेज़ी में ही रखे गए हैं, ताकि आप वही शब्द सीखें जो papers, courses और code में मिलते हैं।

## चलती animations कहाँ हैं

- **`CS229_Hindi_Animated.html`** (इसी folder में): इसे browser में खोलें। यह यही पूरी guide है, पर इसमें सारी animations चलती हैं।
- **`cs229_hindi/out/anim/`**: हर animation की अलग GIF file। नाम `L02_3.gif` का मतलब है Lecture 2 की तीसरी animation।

## Code कैसे चलाएँ

हर chapter का code `cs229_hindi/chapters/<chapter>/example.py` में है। Terminal में इसी folder से:

```
cs229_hindi/.venv/bin/python cs229_hindi/chapters/L02_linear_regression/example.py
```

Animations का code उसी folder की `anims.py` में है। पूरी guide दोबारा बनाने के लिए: `cs229_hindi/.venv/bin/python cs229_hindi/build.py`

## पहले से क्या आना चाहिए

- **Linear Algebra:** vectors, matrices, matrix multiplication, transpose, inverse, eigenvalues / eigenvectors।
- **Probability:** random variable, expectation, variance, conditional probability, Bayes rule, Gaussian distribution।
- **Calculus:** derivative, partial derivative, gradient, chain rule।
- **Python:** NumPy arrays की बुनियादी समझ।

## पूरे course में चलने वाला notation

| Symbol | मतलब |
|---|---|
| $x^{(i)}, y^{(i)}$ | $i$-वाँ training example: input और target |
| $m$ | training examples की संख्या |
| $n$ | features की संख्या (या input का dimension) |
| $\theta$, $w$, $b$ | model के parameters |
| $h_\theta(x)$ | hypothesis: model का prediction |
| $J(\theta)$ | cost function जिसे minimize करना है |
| $\ell(\theta)$ | log-likelihood जिसे maximize करना है |
| $\alpha$ | learning rate |
| $\nabla_\theta$ | $\theta$ के सापेक्ष gradient |
| $\mathcal{N}(\mu, \Sigma)$ | mean $\mu$ और covariance $\Sigma$ वाला Gaussian |
| $:=$ | "दाएँ की value बाएँ में रख दो" (assignment) |
