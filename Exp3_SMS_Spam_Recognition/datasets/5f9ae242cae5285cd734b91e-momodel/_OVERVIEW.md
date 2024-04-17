### 背景
垃圾短信 (Spam Messages，SM) 是指未经过用户同意向用户发送不愿接收的商业广告或者不符合法律规范的短信。

随着手机的普及，垃圾短信在日常生活日益泛滥，已经严重的影响到了人们的正常生活娱乐，乃至社会的稳定。

据 360 公司 2020 年第一季度有关手机安全的报告提到，360 手机卫士在第一季度共拦截各类垃圾短信约 34.4 亿条，平均每日拦截垃圾短信约 3784.7 万条。

大数据时代的到来使得大量个人信息数据得以沉淀和积累，但是庞大的数据量缺乏有效的整理规范；
在面对量级如此巨大的短信数据时，为了保证更良好的用户体验，如何从数据中挖掘出更多有意义的信息为人们免受垃圾短信骚扰成为当前亟待解决的问题。
### 内容
本数据集主要包含2部分，`sms_pub.csv` 包括约 7 万条数据，每条数据 有 3 个字段 label、 message 和 msg_new， 分别代表了短信的类别、短信的内容和分词后的短信；而短信 label  分为正常短息和恶意短信另种类别，示意如下：

<img src="https://imgbed.momodel.cn/20201029224123.png" width=400px/>

scu_stopwords.txt 是四川大学机器智能实验室停用词库，网址：https://github.com/goto456/stopwords/blob/master/scu_stopwords.txt

### 致谢
本数据集是自动收集获取，如有雷同，请联系我们！

