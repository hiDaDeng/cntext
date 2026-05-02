# 四、词向量资源

## 4.1 预训练模型

使用 cntext2.x 训练得到的中文预训练模型资源，汇总如下

对中文语料进行了近义测试和类比测试， 其中斯皮尔曼秩系数(Spearman's Rank Coeficient) 取值[-1,1], 取值越大表示模型越符合人类的认知。

类比测试有首都国家（CapitalOfCountries）、省会省份（CityInProvince）、家人关系（FamilyRelationship）、社会科学(管理、经济、心理等 SocialScience) 的类别准确率测试。

<br>

| 数据集                                                                                              | 词向量                                                                                                                    | 网盘                                                      | 斯皮尔曼秩系数 | 首都国家(%) | 省会省份(%) | 家人关系(%) | 社会科学(%) |
| --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | -------------- | ----------- | ----------- | ----------- | ----------- |
| [中国政府工作报告](https://textdata.cn/blog/2023-12-17-gov-anual-report-dataset/)                   | **_人民政府(国省市)工作报告-GloVe.200.15.bin_**                                                                           | https://pan.baidu.com/s/1IdK8RU9L8mp6I2nhcoSmyA?pwd=ht2s  | 0.38           | 30.73       | 98.86       | 0.00        | 0.00        |
| [中国政府工作报告](https://textdata.cn/blog/2023-12-17-gov-anual-report-dataset/)                   | **_人民政府(国省市)工作报告-Word2Vec.200.15.bin_**                                                                        | https://pan.baidu.com/s/1GoTjMbUcYS4jN6w4GqlqBA?pwd=qb5b  | 0.35           | 30.06       | 96.00       | 0.00        | 16.67       |
| [中国裁判文书网](https://textdata.cn/blog/2023-05-07-china-law-judgment-documents-datasets/)        | **_裁判文书-GloVe.200.15.bin_**                                                                                           | https://pan.baidu.com/s/1a0Fisvnkl8UaQZrHP7olCQ?pwd=8w49  | 0.37           | 7.69        | 98.86       | 75.53       | 25.00       |
| [留言板](https://textdata.cn/blog/2023-12-22-renmin-gov-leader-comment-board/)                      | **_留言板-Word2Vec.200.15.bin_**                                                                                          | https://pan.baidu.com/s/1n7vwCOBnrye1CYrt_IBqZA?pwd=9m42  | 0.45           | 19.33       | 100         | 61.40       | 20          |
| [留言板](https://textdata.cn/blog/2023-12-22-renmin-gov-leader-comment-board/)                      | **_留言板-GloVe.200.15.bin_**                                                                                             | https://pan.baidu.com/s/1e5Y5enOaSUsBdkpg8byWbw?pwd=8zg7  | 0.38           | 12.61       | 100         | 65.81       | 25.00       |
| [A 股年报](https://textdata.cn/blog/2023-03-23-china-a-share-market-dataset-mda-from-01-to-21/)     | **_mda01-24-GloVe.200.15.bin_**   | https://pan.baidu.com/s/1TqoA4TqMAhLzpIp0ZvrQEA?pwd=ajjw  | 0.35          | 77.14      | 100         | 0           | 25.0       |
| [A 股年报](https://textdata.cn/blog/2023-03-23-china-a-share-market-dataset-mda-from-01-to-21/)     | **_mda01-24-Word2Vec.200.15.bin_**     | https://pan.baidu.com/s/1Gke4UKOnswpctp8vsZ0koQ?pwd=dpry  | 0.42   | 31.21      | 97.71       | 10          | 44.44       |
| [港股年报](https://textdata.cn/blog/2024-01-21-hk-stock-market-anual-report/)                       | **_英文港股年报-Word2Vec.200.15.bin_**                                                                                    | https://pan.baidu.com/s/1ISGAoZnA_1Ben6M2DCliOQ?pwd=nagx  | ---            | ---         | ---         | ---         | ---         |
| [港股年报](https://textdata.cn/blog/2024-01-21-hk-stock-market-anual-report/)                       | **_中文港股年报-Word2Vec.200.15.bin_**                                                                                    | hhttps://pan.baidu.com/s/1smMcrPtIP8g635YABCodig?pwd=sjdj | 0.35           | 25.20       | 79.43       | 18.59       | 25          |
| [人民日报](https://textdata.cn/blog/2023-12-14-daily-news-dataset/)                                 | [年份 Word2Vec](https://textdata.cn/blog/2023-12-28-visualize-the-culture-change-using-people-daily-dataset/)             | https://pan.baidu.com/s/1Ru_wxu9egsmhM7lATjSlgQ?pwd=bcea  |                |             |             |             |             |
| [人民日报](https://textdata.cn/blog/2023-12-14-daily-news-dataset/)                                 | [对齐模型 Aligned_Word2Vec](https://textdata.cn/blog/2023-12-28-visualize-the-culture-change-using-people-daily-dataset/) | https://pan.baidu.com/s/1IVgP0MyQpez0hpoJyEyFdA?pwd=7qsu  |                |             |             |             |             |
| [专利申请](https://textdata.cn/blog/2023-04-13-3571w-patent-dataset-in-china-mainland/)             | **_专利摘要-Word2Vec.200.15.bin_**                                                                                        | https://pan.baidu.com/s/1FHI_J7wU9eQGRckD12QB5g?pwd=6rr2  | 0.46           | 3.78        | 25.14       | 33.33       | 37.50       |
| [专利申请](https://textdata.cn/blog/2023-11-20-word2vec-by-year-by-province/)                       | **_province_w2vs 分省份训练词向量_**                                                                                      | https://pan.baidu.com/s/1eBFTIZcv2DWssLiaRnCqZQ?pwd=ikpu  |                |             |             |             |             |
| [专利申请](https://textdata.cn/blog/2023-11-20-word2vec-by-year-by-province/)                       | **_year_w2vs 分年份训练词向量_**                                                                                          | https://pan.baidu.com/s/1lrVkML92cVJdHQa1HQyAwA?pwd=4gqa  |                |             |             |             |             |
| 大众点评评论语料                                                                                    | **_大众点评-评论-Word2Vec.200.15.bin_**                                                                                   | https://pan.baidu.com/s/15He728XGzoXDFYrUWDTaqQ?pwd=eg6x  | 0.34           | 50.31       | 89.71       | 70.00       | 0.00        |
| 大众点评评论语料                                                                                    | **_大众点评-评论-GloVe.200.15.bin_**                                                                                      | https://pan.baidu.com/s/1cKyv0-CuMqnuM2ENElF6rw?pwd=2b44  | 0.36           | 55.83       | 86.29       | 94.29       | 0.00        |
| 中文歌词                                                                                            | **_中文歌词-Word2Vec.200.15.bin_**                                                                                        | https://pan.baidu.com/s/1h1g1mOACmpCwn5pz8jR3vQ?pwd=ub2z  | 0.06           | 0.00        | 0.00        | 0.9         | 0.00        |
| 英文歌词                                                                                            | **_英文歌词-Word2Vec.200.15.bin_**                                                                                        | https://pan.baidu.com/s/1ycy-BTSa8zqW_xbIoshy6Q?pwd=hu1v  |                |             |             |             |             |
| [黑猫消费者投诉](https://textdata.cn/blog/2025-03-05-consumer-complaint-dataset/)                   | **_消费者黑猫投诉-Word2Vec.200.15.bin_**                                                                                  | https://pan.baidu.com/s/1FOI2BIVRojOswdKfqaNbsw?pwd=catc  | 0.32           | 16.18       | 68          | 28.57       | 0.00        |
| [豆瓣影评](https://textdata.cn/blog/2024-04-16-douban-movie-1000w-ratings-comments-dataset)         | **_douban-movie-1000w-Word2Vec.200.15.bin_**                                                                              | https://pan.baidu.com/s/1uq6Ti7HbEWyT4CgktKrMng?pwd=63jg  | 0.43           | 39.02       | 28.57       | 92.65       | 25.00       |
| [B 站](https://textdata.cn/blog/2023-11-12-using-100m-bilibili-user-sign-data-to-training-word2vec) | **_B 站签名-Word2Vec.200.15.bin_**                                                                                        | https://pan.baidu.com/s/1OtBU9BzitcNxkmPzhzH6FQ?pwd=m3iv  | 0.34           | 25.56       | 33.71       | 44.17       | 0.00        |
| [B 站弹幕](https://github.com/Viscount/IUI-Paper)                                                   | **_B 站弹幕-Word2Vec.200.15.bin_**                                                                                        | https://pan.baidu.com/s/1LNDLed5uP3KnUMmrKf_uhg?pwd=x4t8  | 0.42           | 11.67       | 65.81       | 44.17       | 25.00       |

<br>

如使用以上预训练模型发表论文，可在论文中添加[引用信息](https://cntext.readthedocs.io/zh-cn/latest/cite.html)。

在论文方法部分，感兴趣的可以文本如"经过数据采集、语料构建， 使用 cntext 库 GloVe（或 Word2Vec）算法，将参数设置为窗口(window)15，维度数 200， 训练得到预训练模型。 经过近义词测试、类比测试， 模型效果良好。"

<br><br>

## 4.2 相关代码

- [实验 | 使用 Stanford Glove 代码训练中文语料的 GloVe 模型](https://textdata.cn/blog/2025-03-28-train_a_glove_model_on_chinese_corpus_using_stanfordnlp/)
- [词向量 | 使用**人民网领导留言板**语料训练 Word2Vec 模型](https://textdata.cn/blog/2023-12-28-train-word2vec-using-renmin-gov-leader-board-dataset/)
- [可视化 | 人民日报语料反映七十年文化演变](https://textdata.cn/blog/2023-12-28-visualize-the-culture-change-using-people-daily-dataset/)
- [使用 5000w 专利申请数据集按年份(按省份)训练词向量](https://textdata.cn/blog/2023-11-20-word2vec-by-year-by-province/)

<br><br>

## 4.3 相关文献

- [大数据时代下社会科学研究方法的拓展——基于词嵌入技术的文本分析的应用](https://textdata.cn/blog/2022-04-07-word-embeddings-in-social-science/)
- [OS2022 | 概念空间 | 词嵌入模型如何为组织科学中的测量和理论提供信息](https://textdata.cn/blog/2023-11-03-organization-science-with-word-embeddings/)
- [词嵌入技术在社会科学领域进行数据挖掘常见 39 个 FAQ 汇总](https://textdata.cn/blog/2023-03-15-39faq-about-word-embeddings-for-social-science/)
