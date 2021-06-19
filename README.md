# 多元迴歸分析－解析財務報表

```
課程名稱：東吳大學 財精系選修課－迴歸分析
時間：2021 Spring
指導老師：蕭維政 助理教授

團隊成員：
1. 江祐宏
2. 陳亮妘
3. 蔡其浩
```

## 一、緒論
一間公司經營的好或壞，都會被會計師全面地揭露於財務報表之中。這些精準的量化數據反映了一整年的獲利與營運情況，其中包含了「資產負債面」、「現金流量」、「年度損益」等等。然而，是誰會去閱覽這些資訊呢？除了政府機關作為報稅依據外，莫過於市場上的投資者了。

股票交易的一個派別，價值投資者認為，一間公司的經營狀況決定了該公司股票的真實價值。他們會去持有財務優良、獲利健全的公司，並認為這些公司的價值會持續增長。換個說法，假設市場長期來看是有效率的，意即當發現價格對於價值出現低估時，市場上的投資者就會買入，直到價格恢復均衡，反之亦然。

正因如此，我們可以說，在不考慮總體經濟局勢下（例如：金融海嘯時股市全面崩盤），財務報表的資訊應是該年股價變動很大的一個關鍵。每三個月所發佈的財務數據，或說內部人先知財務數據的好壞，應會即時反映在股價走勢上。因此，我們決定驗證此假說，試著以「隔年 3 月」發佈的年度財務報表，解讀當年（1 月至 12 月）的股票報酬率；而以一整年的資料作為分析的原因，為的是排除短期內股價不均衡的波動，避免影響模型分析結果。

## 二、實作流程
#### 1. 資料蒐集與分析
使用「東吳大學 TEJ pro 資料庫系統」，下載 2009 ~ 2020 年底的上市公司股價資料與年度累積財報數據。完成資料清洗與彙整後，透過 `pandas-profiling` 套件快速製作出 [探索性資料分析報表](https://alexchiang0208.github.io/RegressionAnalysis-Annual-Return-and-Financial-Index/Report/EDA_report.html)。

#### 2. 原始模型
將 45+19 個變數建立原始迴歸模型，作為之後的比較報表。接著將連續變數做共線性檢定，刪除 VIF 值大於 10 的變數。

* 45 個連續型解釋變數
* 1 個分類變數（上市公司產業別） -> 轉換成 19 個虛擬變數

#### 3. 模型的選取方式
使用向前選取法、向後選取法、逐步選取法，挑選出精簡的模型。經過 F 檢定，得知模型具有顯著性，但解釋能力較低，可再做優化。

#### 4. 殘差檢定
檢驗殘差的三大假設－常態性、獨立性、變異數同質性，發現三者皆未通過，因此，我們考慮增加變數或是做變數轉換。假設做「變數轉換」或「主成份分析」皆有機會解決殘差未過的問題，但由於本次專題的目的在於「找出影響股價較大的財務指標」，因此不考慮將原始數據轉換（無法解讀），而是選擇將變數做交互作用，增加更多的變數建立模型。

#### 5. 再建模型
將「通過共線性檢定的連續變數」全數做交互作用（26 + C26 取 2）與原始的 19 個虛擬變數－共計 370 個變數 再建模型，並使用向前選取法挑選變數，找出精簡有效的模型，最後再做殘差檢定。

## 三、結論
最終結果顯示，若是以嚴謹的統計觀點來看待這個假說，是不足以成立的。不論是在原始模型、刪除共線性過大的變數，或是在加入交互作用變數的新模型，在殘差三大檢定中都未過標準，即便 F 檢定有通過，但我們仍然不能說這是一個精準的模型。

然而，這份報告依舊可以告訴我們一些觀點。雖然在最終的統計檢定未通過，但可以看出，經過一次次改良的模型，調整後的 R-square 都相較原始複迴歸模型來得高，表示加入交互作用的變數之後，對於 Y 的影響有所提升。

我們也能從最終模型的結果，看得出哪些變數對 Y 的影響較大，這也是我們最想關心的議題。觀察報告第六章之－向前選取法，可以發現有 17 個變數的顯著水準為三顆星星，代表這些變數對 Y 的影響較大。假如未來在做投資規劃、股票選擇、財報分析，即可觀察這 17 個變數作為參考依據。

## 四、專題成果

### [專題書面報告](Report/書面報告.pdf)
### [資料分析報表](https://alexchiang0208.github.io/RegressionAnalysis-Annual-Return-and-Financial-Index/Report/EDA_report.html)


## 五、程式碼
1. [TEJ 資料清洗](TEJ_data_clearing.py)
2. [探索性資料分析](EDA_report.ipynb)
3. [原始迴歸模型與 VIF 檢定](origin_model.py)
4. [迴歸分析技巧](regression_analysis.R)

---

## 學習筆記
在做迴歸分析之前，最重要的一件事就是要清楚問題是什麼、要解決什麼事情。因為以下兩種情境作法完全不一樣，甚至可能互相矛盾，因此定義好做分析的**目的**是非常必要的。

#### 解釋性建模
1. 假設 X1 = X2，那麼 Y = X1 + X2 和 Y = 2 * X2 和 Y = 3 * X1 - X2 這三個一樣的公式，在解讀上卻完全不一樣。在做解釋性建模時，首先要解決共線性的問題。
2. 把原始資料做過太多次轉換會變得難以解釋，不建議數值轉換，除非是實務上的經驗或是真的有所根據。例如：因為殘差檢定沒過而把原始數據加了一個很大的常數、取平方、開根號，甚至做了 PCA，會變得無法解讀，直不直覺比較重要。
3. 學術上重視統計檢定是否通過、滿足嚴謹的數學假設；但在實務上，很難做到所有（殘差）檢定都滿足，而是會注重在如何處理分類變數（分類變數影響截距/斜率/都影響、轉換成虛擬變數），並達到直覺解釋的目的。
5. 在做變數的選取法（根據每個變數的顯著性）時，要有一個概念「Xi 的工作能力，是會隨著同事不同而有所不同的」，這說明了逐步選取法的優勢，因為刪除後的變數能夠敗步復活。
6. 比 F 檢定更好判斷模型優劣的準則＝》AIC: 選中最有效，但較為複雜的模型；BIC: 選中最精簡，具有一致性的有效模型。

#### 預測性建模
1. 重點放在「對未來資料的預測效果」。模型的 R-square、解釋變數的假設、共線性問題、殘差檢定等，**通通都不重要！**只要能夠讓預測效果變好，就都是好方法。
2. 將資料切割成 Train Data（建模用）以及 Test Data（預測用）。Train Data 如何都不重要，我們只關心 Test Data。
3. 會將變數做 PCA/Lasso/Ridge，就是為了讓預測效果變好。我們發現，Python `sklearn` 套件的迴歸方法，可以做 PCA/Lasso/Ridge/CV，卻從來不會用它去做殘差檢定、共線性檢定，因為它根本沒有，也不需要這些功能（做解釋性的統計方法在 Python 上推薦 `statsmodels`；做預測性/機器學習則是推薦 `sklearn`，兩者解決不同的問題）。
