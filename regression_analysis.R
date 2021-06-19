# 多元回歸
library(readr)

Reg_XY_data <- read_csv("/Users/alex_chiang/Documents/GitHub/RegressionAnalysis-Annual-Return-and-Financial-Index/Tidy_Data/reg_XY_new.csv")
Reg1 <- subset(Reg_XY_data, select = c(-Time,-stock_id))

# 建立虛擬變數(Dummy Variable)
library(fastDummies)
Reg2 <- dummy_cols(Reg1 , select = "交易所主產業代碼",
                   remove_selected_columns = TRUE)

# 變數簡稱
colnames(Reg2) <- c("x1","x2","x3","x4","x5","x6","x7","x8","x9","x10",
                   "x11","x12","x13","x14","x15","x16","x17","x18","x19","x20",
                   "x21","x22","x23","x24","x25","x26","x27","x28","x29","x30",
                   "x31","x32","x33","x34","x35","x36","x37","x38","x39","x40",
                   "x41","x42","x43","x44","x45","y","d1","d2","d3","d4","d5","d6","d7",
                   "d8","d9","d10","d11","d12","d13","d14","d15","d16","d17","d18","d19")

# 刪除多元共線性變數（From python information）
Reg_del.vif <- subset(Reg2,select = c(-x1,-x4,-x5,-x6,-x7,-x8,-x10,
                                      -x11,-x16,-x17,-x21,-x22,-x27,
                                      -x29,-x30,-x31,-x33,-x34,-x36))

# 模型變項選取
fullmodel = lm(y~.,data=Reg_del.vif)
summary(fullmodel)

nullmodel = lm(y~1,data=Reg_del.vif)
summary(nullmodel)

## 向前選取法(forward selection method)
forward.lm = step(nullmodel,scope = list(lower=nullmodel,upper=fullmodel),
                  direction = "forward")
summary(forward.lm)

## 向後選取法(backward selection method)
backward.lm = step(fullmodel,scope = list(upper=fullmodel),
                   direction = "backward")
summary(backward.lm)

## 逐步選取法(stepwise selection method)
library(MASS)
stepwise.lm=stepAIC(fullmodel,direction="both")
summary(stepwise.lm)

# 模型選擇的變數
Reg <- subset(Reg_del.vif,select = c(-x9,-x12,-x13,-x14,-x18,-x19,-x20,
                                     -x23,-x32,-x35,-x37,-x39,-x40,-x42,-x44,
                                     -x45,-d1,-d2,-d3,-d6,-d7,-d8,-d9,-d10,
                                     -d11,-d14,-d15,-d17,-d18,-d19))
mlm1 <- lm(y~.,data=Reg)
summary(mlm1)

# 去除p-value > 0.1的變數
Reg_final <- subset(Reg,select = c(-x25,-x28))
mlm <- lm(y~.,data=Reg_final)
summary(mlm)


# 複迴歸加入連續變數的交互作用項
mlm_interact = lm(y~(x2+x3+x9+x12+x13+x14+x15+x18+x19+x20+
                     x23+x24+x25+x26+x28+x32+x35+x37+x38+
                     x39+x40+x41+x42+x43+x44+x45)*(
                     x2+x3+x9+x12+x13+x14+x15+x18+x19+x20+
                     x23+x24+x25+x26+x28+x32+x35+x37+x38+
                     x39+x40+x41+x42+x43+x44+x45)+(
                     d1+d2+d3+d4+d5+d6+d7+d8+d9+d10+d11+d12+d13+
                     d14+d15+d16+d17+d18+d19),data= Reg_del.vif)

options(max.print = 10000)
summary(mlm_interact)

# 模型變項選取
fullmodel = lm(y~(x2+x3+x9+x12+x13+x14+x15+x18+x19+x20+
                  x23+x24+x25+x26+x28+x32+x35+x37+x38+
                  x39+x40+x41+x42+x43+x44+x45)*(
                  x2+x3+x9+x12+x13+x14+x15+x18+x19+x20+
                  x23+x24+x25+x26+x28+x32+x35+x37+x38+
                  x39+x40+x41+x42+x43+x44+x45)+(
                  d1+d2+d3+d4+d5+d6+d7+d8+d9+d10+d11+d12+d13+
                  d14+d15+d16+d17+d18+d19),data= Reg_del.vif)
summary(fullmodel)
nullmodel = lm(y~1,data=Reg_del.vif)
summary(nullmodel)

## 向前選取法(forward selection method)
forward.lm = step(nullmodel,scope = list(lower=nullmodel,upper=fullmodel),
                  direction = "forward")
summary(forward.lm)

## 向後選取法(backward selection method)
backward.lm = step(fullmodel,scope = list(upper=fullmodel),
                   direction = "backward")
summary(backward.lm)

## 逐步選取法(stepwise selection method)
library(MASS)
stepwise.lm=stepAIC(fullmodel,direction="both")
summary(stepwise.lm)

# 殘差常態性檢定
library(nortest)
lillie.test(forward.lm$residuals)

# 殘差獨立性檢定
library(car)
durbinWatsonTest(forward.lm)

# 殘差變異數同質性檢定
library(lmtest)
ncvTest(forward.lm)

library(zoo)
bptest(mlm)
