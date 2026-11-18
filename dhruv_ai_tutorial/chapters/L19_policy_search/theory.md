# Policy Search, REINFORCE और POMDPs

> **इस lecture में:** Direct policy search का विचार, value-based तरीकों से इसकी तुलना, parameterized stochastic policies (logistic और softmax), expected return objective $J(\theta)$, log-derivative trick से REINFORCE का पूरा derivation (और transition probabilities कैसे कट जाती हैं), एक trajectory से unbiased gradient estimate, baseline से बिना bias जोड़े variance घटाना ($E[\nabla_\theta \log \pi_\theta] = 0$), reward-to-go, PEGASUS का विचार, POMDPs (observations, belief state और Bayes filter update), और policy search कब चुनना चाहिए।

## 1. एक line में idea

अब तक हमने Reinforcement Learning में पहले value function $V^*$ या $Q^*$ सीखा और फिर उस पर $\arg\max_a$ लगाकर policy निकाली। **Direct policy search** इस बीच वाले कदम को छोड़ देता है: आप सीधे एक parameterized policy $\pi_\theta(a \mid s)$ लिखते हैं और parameters $\theta$ को इस तरह बदलते हैं कि कुल expected reward सबसे ज़्यादा हो जाए।

इस lecture का सबसे सुंदर नतीजा **REINFORCE** है। Log-derivative trick की मदद से आप दुनिया के transition model $P_{sa}$ को जाने बिना, सिर्फ़ policy चलाकर मिले trajectories से expected return का **unbiased gradient** निकाल सकते हैं। और अगर दुनिया का असली state छुपा हो (**POMDP**), तो या तो आप observations से **belief state** बनाकर Bayes filter से update करते हैं, या सीधे observations पर policy search चलाते हैं।

## 2. Direct Policy Search: Setup और Stochastic Policies

मान लीजिए state $s \in S$ (या feature vector $\phi(s) \in \mathbb{R}^d$) दिया है और आप $\theta$ से तय होने वाली एक **stochastic policy** $\pi_\theta(a \mid s)$ चुनते हैं। यहाँ $\pi_\theta(a \mid s)$ वह probability है जिससे agent state $s$ में action $a$ लेता है, इसलिए हर $s$ पर $\sum_{a \in A} \pi_\theta(a \mid s) = 1$।

Deterministic policy $\pi(s) \in A$ की जगह stochastic policy क्यों? इसके दो बड़े कारण हैं:

1. **Smoothness और gradient:** अगर policy अचानक एक action से दूसरे पर कूदती है, तो $\theta$ में छोटे बदलाव से discrete छलाँग लगती है और gradient काम नहीं करता। Stochastic policy में $\pi_\theta(a \mid s)$ parameters $\theta$ का smooth और differentiable function होता है, इसलिए expected return $J(\theta)$ पर सीधे **gradient ascent** लगाया जा सकता है।
2. **Exploration और अधूरी जानकारी:** Stochastic policy अपने आप अलग-अलग actions आज़माती है। आगे हम देखेंगे कि POMDPs और कुछ games में सबसे अच्छी policy खुद stochastic हो सकती है।

### दो आम policy परिवार

**(क) Logistic policy (दो actions के लिए)।** जब $A = \{0, 1\}$ हो (जैसे cart-pole में बाएँ या दाएँ धक्का देना) और state features $\phi(s) \in \mathbb{R}^d$ हों, तो आप logistic sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ इस्तेमाल करते हैं:

$$\pi_\theta(a = 1 \mid s) = \sigma(\theta^T \phi(s)) = \frac{1}{1 + e^{-\theta^T \phi(s)}}, \qquad \pi_\theta(a = 0 \mid s) = 1 - \sigma(\theta^T \phi(s))$$

जब $\|\theta\|$ छोटा हो तो दोनों actions लगभग 0.5-0.5 probability से चुने जाते हैं। जैसे-जैसे $\|\theta\|$ बढ़ता है, यही smooth surface तेज़ होकर लगभग deterministic boundary बन जाती है।

**(ख) Softmax policy ($k$ actions के लिए)।** जब $A = \{1, 2, \dots, k\}$ में दो से ज़्यादा actions हों (जैसे gridworld में चार दिशाएँ), तो हर state-action जोड़े के features $\phi(s, a) \in \mathbb{R}^d$ (या हर action का अलग weight vector $\theta_a$) लेकर **softmax** लिखा जाता है:

$$\pi_\theta(a \mid s) = \frac{\exp(\theta^T \phi(s, a))}{\sum_{a' \in A} \exp(\theta^T \phi(s, a'))}$$

यह हर action को धनात्मक probability देता है और जोड़ हमेशा 1 रखता है। इन दोनों परिवारों के log-probabilities के gradients बहुत साफ़ आते हैं, जो आगे REINFORCE में सीधे काम आएँगे।

## 3. Objective Function: Expected Return

मान लीजिए एक episode $T$ कदमों तक चलता है (horizon $T$)। एक पूरी **trajectory** (या rollout) यह अनुक्रम है:

$$\tau = (s_0, a_0, s_1, a_1, \dots, s_{T-1}, a_{T-1}, s_T)$$

इस trajectory का कुल return (total payoff) है:

$$R(\tau) = \sum_{t=0}^{T-1} \gamma^t R(s_t, a_t)$$

Finite-horizon के सीधे derivation में अक्सर $\gamma = 1$ रखकर $R(\tau) = \sum_{t=0}^{T-1} R(s_t, a_t)$ लिखा जाता है; $\gamma < 1$ होने पर भी नीचे का हर कदम बिल्कुल वैसे ही चलता है।

Policy $\pi_\theta$ के तहत किसी खास trajectory $\tau$ के घटने की probability $p_\theta(\tau)$ क्या है? दुनिया पहले initial distribution $\mu(s_0)$ से $s_0$ चुनती है, फिर हर कदम पर agent $\pi_\theta(a_t \mid s_t)$ से $a_t$ चुनता है और दुनिया $P_{s_t a_t}(s_{t+1})$ से अगला state निकालती है:

$$p_\theta(\tau) = \mu(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t)\, P_{s_t a_t}(s_{t+1})$$

हमारा लक्ष्य है expected return को maximize करना:

$$J(\theta) = E_{\tau \sim p_\theta}\!\left[ R(\tau) \right] = \sum_{\tau} p_\theta(\tau)\, R(\tau)$$

(Continuous states या actions हों तो $\sum_\tau$ की जगह $\int d\tau$ आ जाता है, बाकी सब वैसा ही रहता है।) हम $\theta$ को **gradient ascent** से सुधारना चाहते हैं: $\theta := \theta + \alpha \nabla_\theta J(\theta)$। पर मुश्किल यह है कि $p_\theta(\tau)$ के अंदर दुनिया का अज्ञात transition model $P_{sa}$ बैठा है, और सारी संभव trajectories पर जोड़ लेना असंभव है।

## 4. REINFORCE Derivation: Log-Derivative Trick

यहाँ calculus की एक छोटी-सी पर बेहद शक्तिशाली पहचान काम आती है। किसी भी धनात्मक differentiable function $p_\theta(\tau)$ के लिए $\nabla_\theta \log p_\theta(\tau) = \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}$, यानी:

$$\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau)$$

इसे **log-derivative trick** (या likelihood-ratio trick) कहते हैं। इसे $\nabla_\theta J(\theta)$ में लगाएँ:

$$\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \sum_{\tau} p_\theta(\tau)\, R(\tau) \\
&= \sum_{\tau} \left( \nabla_\theta p_\theta(\tau) \right) R(\tau) \\
&= \sum_{\tau} p_\theta(\tau) \left( \nabla_\theta \log p_\theta(\tau) \right) R(\tau) \\
&= E_{\tau \sim p_\theta}\!\left[ R(\tau)\, \nabla_\theta \log p_\theta(\tau) \right]
\end{aligned}$$

अब जादू देखिए: $\log p_\theta(\tau)$ को खोलकर लिखें। गुणनफल $\prod$ अब जोड़ $\sum$ बन जाता है:

$$\log p_\theta(\tau) = \log \mu(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T-1} \log P_{s_t a_t}(s_{t+1})$$

अब दोनों ओर $\theta$ के सापेक्ष gradient $\nabla_\theta$ लें। Initial state distribution $\mu(s_0)$ में $\theta$ नहीं है, और दुनिया की transition probabilities $P_{s_t a_t}(s_{t+1})$ में भी $\theta$ कहीं नहीं है! इसलिए उनके derivatives शून्य हो जाते हैं:

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

इसे वापस ऊपर रखने पर **Policy Gradient Theorem (REINFORCE रूप)** मिलता है:

$$\nabla_\theta J(\theta) = E_{\tau \sim p_\theta}\!\left[ R(\tau) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]$$

इस सूत्र पर रुककर गौर करें: **transition probabilities $P_{sa}$ पूरी तरह कट चुकी हैं!** Gradient निकालने के लिए आपको दुनिया के physics या transition matrices का कोई formula नहीं चाहिए। बस दुनिया में (या simulator में) $\pi_\theta$ चलाकर trajectory sample करना आना चाहिए।

### एक trajectory से Unbiased Estimate और इसका मतलब

क्योंकि $\nabla_\theta J(\theta)$ एक expectation है, अगर आप वर्तमान policy $\pi_\theta$ से सिर्फ़ **एक trajectory** $\tau$ चलाएँ, तो

$$\hat{g}(\tau) = R(\tau) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

सच्चे gradient का एक **unbiased estimate** है, यानी $E_{\tau \sim p_\theta}[\hat{g}(\tau)] = \nabla_\theta J(\theta)$। $m$ trajectories का औसत लेने पर noise और कम हो जाता है।

Update नियम $\theta := \theta + \alpha \hat{g}(\tau)$ का सहज मतलब बहुत साफ़ है:

- अगर trajectory का कुल return $R(\tau) > 0$ (अच्छा) आया, तो हम उन सारे actions $a_t$ की log-probability $\log \pi_\theta(a_t \mid s_t)$ बढ़ाते हैं जो इस trajectory में लिए गए थे।
- अगर $R(\tau) < 0$ (खराब) आया, तो हम उन्हीं actions की probability घटा देते हैं।

**Score function $\nabla_\theta \log \pi_\theta(a \mid s)$ कैसा दिखता है?**

- **Logistic policy** के लिए: $\nabla_\theta \log \pi_\theta(a \mid s) = \left(a - \sigma(\theta^T \phi(s))\right) \phi(s)$।
- **Softmax policy** के लिए:

$$\nabla_\theta \log \pi_\theta(a \mid s) = \phi(s, a) - \sum_{a' \in A} \pi_\theta(a' \mid s)\, \phi(s, a') = \phi(s, a) - E_{a' \sim \pi_\theta(\cdot \mid s)}\!\left[\phi(s, a')\right]$$

यानी चुने गए action के features की तरफ़ बढ़ो, और औसत features से दूर हटो।

## 5. Variance की समस्या, Baseline और Reward-to-Go

अकेला REINFORCE unbiased तो है, पर उसका **variance** बहुत बड़ा हो सकता है। एक आसान उदाहरण सोचिए: मान लीजिए हर trajectory का return धनात्मक है, जैसे कुछ trajectories में $R(\tau) = +98$ मिलता है और कुछ में $R(\tau) = +102$। बिना किसी सुधार के $R(\tau)$ दोनों में धनात्मक है, इसलिए update **हर** trajectory के actions की probability ऊपर धकेलने की कोशिश करता है (एक को 98 गुना, दूसरे को 102 गुना)। इससे अलग-अलग trajectories के gradients बेतहाशा इधर-उधर भागते हैं। हमें असल में यह चाहिए था कि औसत ($100$) से ऊपर वाली trajectory को बढ़ावा मिले और $100$ से नीचे वाली को दबाया जाए।

### Baseline: बिना Bias जोड़े Variance घटाना

क्या हम $R(\tau)$ में से कोई संख्या $b$ (**baseline**) घटा सकते हैं, और फिर भी gradient unbiased रहे? हाँ! इसकी वजह यह मूलभूत पहचान है कि किसी भी state $s$ पर score function का expectation शून्य होता है:

$$\begin{aligned}
E_{a \sim \pi_\theta(\cdot \mid s)}\!\left[ \nabla_\theta \log \pi_\theta(a \mid s) \right]
&= \sum_{a \in A} \pi_\theta(a \mid s)\, \frac{\nabla_\theta \pi_\theta(a \mid s)}{\pi_\theta(a \mid s)} \\
&= \sum_{a \in A} \nabla_\theta \pi_\theta(a \mid s) = \nabla_\theta \left( \sum_{a \in A} \pi_\theta(a \mid s) \right) = \nabla_\theta (1) = 0
\end{aligned}$$

इसलिए किसी भी constant $b$ के लिए (या किसी भी ऐसे baseline $b(s_t)$ के लिए जो उस कदम के action $a_t$ पर निर्भर नहीं करता), iterated expectation लगाने पर:

$$E_{\tau \sim p_\theta}\!\left[ b(s_t)\, \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] = E_{s_t}\!\left[ b(s_t)\, E_{a_t \sim \pi_\theta(\cdot \mid s_t)}\!\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] \right] = 0$$

अतः $b$ घटाने से expectation बिल्कुल नहीं बदलता:

$$\nabla_\theta J(\theta) = E_{\tau \sim p_\theta}\!\left[ (R(\tau) - b) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]$$

Bias शून्य रहता है, पर जब $b \approx E[R(\tau)]$ (जैसे पिछले returns का औसत, या सीखा हुआ value estimate $\hat{V}(s_t)$) चुना जाता है, तो $(R(\tau) - b)$ शून्य के आस-पास केंद्रित रहता है और gradient का variance नाटकीय रूप से गिर जाता है।

### Reward-to-Go (Causality)

दूसरा बड़ा सुधार **causality** से आता है: समय $t$ पर लिया गया action $a_t$ भूतकाल के rewards $r_0, r_1, \dots, r_{t-1}$ को प्रभावित नहीं कर सकता! गणित में, जब $t' < t$ हो तो $r_{t'}$ पहले ही तय हो चुका है, इसलिए $a_t$ पर conditional expectation लेने पर $E[r_{t'} \nabla_\theta \log \pi_\theta(a_t \mid s_t)] = 0$। यानी पुराने rewards सिर्फ़ बेकार का शोर (noise) जोड़ रहे थे।

उन्हें हटाकर हर कदम $t$ को सिर्फ़ उसके **आगे आने वाले return (reward-to-go)** से गुणा करें:

$$G_t = \sum_{t'=t}^{T-1} \gamma^{t'-t} R(s_{t'}, a_{t'})$$

Baseline $b(s_t)$ और reward-to-go $G_t$ दोनों मिलाने पर व्यावहारिक REINFORCE estimator बनता है:

$$\hat{g} = \sum_{t=0}^{T-1} \left( G_t - b(s_t) \right) \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

जब $b(s_t) = \hat{V}(s_t)$ रखा जाता है, तो $G_t - \hat{V}(s_t)$ ठीक इस बात का अंदाज़ा है कि action $a_t$ इस state के सामान्य औसत से कितना बेहतर निकला (इसे आगे चलकर **advantage** कहते हैं)।

## 6. Algorithm: REINFORCE (Step by Step)

1. Policy parameters $\theta$ को शून्य या छोटे random मानों से शुरू करें। Baseline $b := 0$ रखें।
2. हर iteration में वर्तमान policy $\pi_\theta$ चलाकर $m$ trajectories $\tau^{(1)}, \dots, \tau^{(m)}$ एकत्र करें।
3. हर trajectory $i$ और हर कदम $t = 0, \dots, T-1$ के लिए reward-to-go $G_t^{(i)} = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}^{(i)}$ निकालें।
4. Baseline $b$ को वर्तमान (या चलती हुई औसत) returns से update करें, जैसे $b := \frac{1}{mT} \sum_{i, t} G_t^{(i)}$ (या state-dependent $\hat{V}(s_t)$ fit करें)।
5. Policy gradient का estimate बनाएँ:

$$\hat{g} = \frac{1}{m} \sum_{i=1}^{m} \sum_{t=0}^{T-1} \left( G_t^{(i)} - b \right) \nabla_\theta \log \pi_\theta\!\left(a_t^{(i)} \mid s_t^{(i)}\right)$$

6. Gradient ascent कदम लें: $\theta := \theta + \alpha\, \hat{g}$, और दोहराएँ।

## 7. PEGASUS का संक्षिप्त परिचय

जब आपके पास दुनिया का एक **generative simulator** हो (जैसे autonomous helicopter का physics simulator), तो Andrew Ng और Michael Jordan का **PEGASUS** algorithm variance घटाने का एक और चतुर रास्ता देता है।

Simulator के अंदर सारा randomness असल में random number generator के numbers (जैसे हवा के झोंके $\xi_0, \xi_1, \dots$) से आता है। अगर आप $m$ अलग-अलग random number sequences (scenarios) पहले से **fix** कर दें, तो हर scenario deterministic बन जाता है! तब आपका अनुमानित objective $\hat{J}(\theta) = \frac{1}{m} \sum_{i=1}^m R(\tau(\theta, \xi^{(i)}))$ सीधे $\theta$ का एक निश्चित (non-noisy) function हो जाता है। इससे दो अलग-अलग policies $\theta$ और $\theta'$ की तुलना बिल्कुल उन्हीं हवा के झोंकों पर होती है ("apples-to-apples comparison"), इसलिए evaluation का noise बहुत घट जाता है और deterministic policies पर भी सीधा search चलाया जा सकता है।

## 8. POMDPs: जब असली State छुपा हो

अब तक हमने माना था कि agent को हर पल पूरा Markov state $s_t$ साफ़ दिखता है। असली दुनिया में ऐसा कम ही होता है: एक robot को अपनी सटीक $(x, y)$ जगह नहीं दिखती, सिर्फ़ camera या laser sensor का शोर-भरा signal मिलता है; एक डॉक्टर को बीमारी का असली स्तर नहीं दिखता, सिर्फ़ test reports दिखती हैं।

इस ढाँचे को **Partially Observable Markov Decision Process (POMDP)** कहते हैं। यह tuple $(S, A, O, \{P_{sa}\}, \{O_s\}, \gamma, R)$ है:

| Symbol | मतलब |
|---|---|
| $S, A, \{P_{sa}\}, \gamma, R$ | अंदर चल रहा साधारण MDP (असली state $s_t$ agent से छुपा है) |
| $O$ | संभव **observations** का set |
| $O_s(o) = P(o_t = o \mid s_t = s)$ | observation model: असली state $s$ में होने पर observation $o$ दिखने की probability |

क्योंकि अकेला वर्तमान observation $o_t$ Markov नहीं है (एक जैसे दिखने वाले दो गलियारे अलग जगहों पर हो सकते हैं), सिर्फ़ $o_t$ देखकर सबसे अच्छा फ़ैसला हमेशा नहीं लिया जा सकता। पूरे इतिहास $(o_1, a_0, o_2, \dots, a_{t-1}, o_t)$ को संक्षेप में रखने का पर्याप्त (sufficient) तरीका है **belief state** $b_t$:

$$b_t(s) = P(s_t = s \mid o_1, \dots, o_t, a_0, \dots, a_{t-1})$$

यानी $b_t$ सारे states $S$ पर एक probability distribution है ($b_t(s) \ge 0$ और $\sum_{s \in S} b_t(s) = 1$), जो बताता है कि अब तक के सारे सुरागों के बाद agent को अपने कहाँ होने का कितना विश्वास है।

### Bayes Filter से Belief Update

जब agent के पास पिछला belief $b_{t-1}$ हो, वह action $a_{t-1}$ ले, और नया observation $o_t$ देखे, तो नया belief $b_t$ दो साफ़ चरणों में निकलता है (**Bayes filter**):

**चरण 1 (Prediction / Motion update):** Action $a_{t-1}$ लेने से संभावनाएँ transition model के अनुसार आगे खिसकती और थोड़ा फैलती हैं:

$$\bar{b}_t(s') = \sum_{s \in S} P_{s, a_{t-1}}(s')\, b_{t-1}(s)$$

**चरण 2 (Measurement / Bayes update):** अब नया observation $o_t$ आता है। जिन states $s'$ में $o_t$ दिखना ज़्यादा संभव है उनका वज़न बढ़ता है, और बाकी का घटता है:

$$b_t(s') = \frac{O_{s'}(o_t)\, \bar{b}_t(s')}{\sum_{s'' \in S} O_{s''}(o_t)\, \bar{b}_t(s'')} \;\propto\; O_{s'}(o_t) \sum_{s \in S} P_{s, a_{t-1}}(s')\, b_{t-1}(s)$$

हर नए सुराग के साथ अनिश्चितता घटती है और belief distribution असली state के आस-पास पैना (sharp) होता जाता है।

सैद्धांतिक रूप से belief state $b_t$ खुद एक नया Markov state बन जाता है (जिसे **belief MDP** कहते हैं)। पर ध्यान दें: अगर $|S| = 100$ discrete states भी हों, तो $b_t$ एक 99-dimensional continuous simplex में रहता है! उस उच्च-आयामी continuous space पर exact $V^*(b)$ निकालना (Value Iteration जैसा काम) लगभग असंभव हो जाता है। यहीं **policy search** बहुत काम आता है: आप या तो belief के कुछ सरल features $\phi(b_t)$ पर, या हाल के कुछ observations की छोटी खिड़की $\phi(o_t, o_{t-1})$ पर सीधे $\pi_\theta(a_t \mid \cdot)$ लगाकर REINFORCE या PEGASUS से अच्छी policy सीख सकते हैं।

## 9. Policy Search कब चुनें?

| परिस्थिति | बेहतर चुनाव और कारण |
|---|---|
| कम discrete states व कम actions, पूरा state दिखता हो | **Value-based** (Value Iteration, Q-learning): Bellman structure का पूरा फ़ायदा लेता है और अक्सर कम samples में सीखता है। |
| **Continuous या high-dimensional actions** (जैसे robot के 7 जोड़ों के torques) | **Policy search**: हर कदम पर $\max_a Q(s, a)$ ढूँढना महँगा या कठिन है, पर $\pi_\theta(a \mid s)$ से action सीधे sample हो जाता है। |
| **POMDP / अधूरी या शोर-भरी जानकारी** | **Policy search**: पूरा state न होने से Bellman backup टूट सकता है, पर REINFORCE में $\nabla_\theta J(\theta)$ का derivation observations पर बनी stochastic policy के लिए भी सही रहता है (या सरल stochastic policy फँसने से बचाती है)। |
| **सरल policy, जटिल value function** | **Policy search**: कई control tasks (जैसे pole balancing या helicopter hover) में अच्छी policy सिर्फ़ चंद linear gains से लिखी जा सकती है, जबकि पूरे state space का $V^*$ बहुत टेढ़ा-मेढ़ा होता है। |

## 10. Practical बातें और आम गलतियाँ

- **बिना baseline के धनात्मक rewards पर REINFORCE चलाना:** अगर सारे rewards $> 0$ हों और आप baseline न घटाएँ, तो gradient variance बहुत बड़ा होगा और सीखना बेहद धीमा या अस्थिर हो जाएगा। हमेशा कम से कम running-mean baseline या $\hat{V}(s_t)$ घटाएँ।
- **Step size $\alpha$ बहुत बड़ा रखना:** Supervised learning में खराब step सिर्फ़ parameters बिगाड़ता है, data वही रहता है। Policy search में एक बहुत बड़ा step policy को खराब कर देता है, जिससे अगले iteration का सारा नया data भी कचरा आता है और policy उबर नहीं पाती (**policy collapse**)।
- **Policy का जल्दी deterministic हो जाना:** अगर softmax के logits बहुत बड़े हो जाएँ ($\pi_\theta \approx 1$), तो agent नया कुछ आज़माना बंद कर देता है और local optimum में फँस जाता है। ज़रूरत पड़ने पर छोटा entropy bonus या मध्यम $\alpha$ मदद करता है।
- **Batch में returns को normalize करना:** व्यवहार में $(G_t - \bar{G}) / (\text{std}(G) + \epsilon)$ इस्तेमाल करने से अलग-अलग reward scales पर भी step size संतुलित रहता है।
- **Bayes filter में zero probability का जाल:** POMDP में अगर किसी सेंसिंग मॉडल में $O_s(o) = 0$ बिल्कुल सख़्त रख दिया जाए, तो एक बार का अजीब sensor glitch सच्चे state की belief को हमेशा के लिए शून्य कर सकता है। इसलिए हर observation में थोड़ी न्यूनतम noise floor ($\epsilon > 0$) रखें।

## 11. खुद से पूछें

1. REINFORCE के derivation में जब हम $\nabla_\theta \log p_\theta(\tau)$ निकालते हैं, तो दुनिया की transition probabilities $P_{s_t a_t}(s_{t+1})$ क्यों गायब हो जाती हैं, और इसका practical फ़ायदा क्या है?
2. अगर किसी problem में हर trajectory के कुल return में $+1000$ जोड़ दिया जाए, तो बिना baseline वाले REINFORCE के expectation और variance पर क्या असर पड़ेगा? सही baseline $b$ यह असर कैसे खत्म करता है?
3. मान लीजिए एक अंधेरे गलियारे में दो दरवाज़े बिल्कुल एक जैसे दिखते हैं। एक ही observation $o_t$ देखने वाली memoryless deterministic policy वहाँ क्यों अटक सकती है, जबकि Bayes filter वाला belief state $b_t$ या stochastic policy कैसे बाहर निकल सकती है?
