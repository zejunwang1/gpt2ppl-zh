gpt2ppl-zh: 基于中文 GPT2 预训练模型的语句困惑度计算
====================================================

本项目展示了使用中文 GPT2 预训练模型实现文档/语句的困惑度计算。

.. code:: shell

   pip install gpt2ppl-zh

或者

.. code:: shell

   pip install git+https://github.com/zejunwang1/gpt2ppl-zh

句子困惑度计算示例可参考 gpt2ppl/demo_sent.py:

.. code:: python

   # coding=utf-8

   import numpy as np
   from gpt2ppl import GPT2PPL

   def main():
       sent1 = "机器学习与人工智能"
       sent2 = "机器学习与人工只能"

       model = GPT2PPL()
       """
       model = GPT2PPL(
           model_name_or_path="uer/gpt2-chinese-cluecorpussmall",
           tokenizer_mode="bert",
           device="cuda",
           stride=512
       )
       """

       ppl1 = model.get_ppl(sent1)
       ppl2 = model.get_ppl(sent2)

       print("Sentence: {}, Perplexity Score: {}".format(sent1, ppl1))
       print("Sentence: {}, Perplexity Score: {}".format(sent2, ppl2))

   if __name__ == "__main__":
       main()

运行结果：

::

   Sentence: 机器学习与人工智能, Perplexity Score: 5.058856964111328
   Sentence: 机器学习与人工只能, Perplexity Score: 74.32606506347656

文档困惑度计算示例可参考
gpt2ppl/demo_doc.py，程序会计算文档中每个句子的困惑度：

.. code:: python

   result = model.get_ppl_per_sentence(text, min_length=8)
   """
   result = model.get_ppl_per_sentence(
       text,
       ratio=0.4, 
       min_length=8, 
       max_length=510, 
       return_loc=False
   )
   min_length: 分句的最小句子长度
   max_length: 分句的最大句子长度
   ratio: 句子中汉字字符的最小比例，若低于该阈值，则句子不参与困惑度计算
   return_loc: 是否返回每个句子在原始文档中的字符位置
   """

运行结果如下：

::

   The Perplexity Score of the whole text is:  5.978885650634766
   The Perplexity Score for single sentences:
   ('中文分词是一种将连续的中文文本划分成有意义的词汇序列的自然语言处理技术。', 17.03954315185547)
   ('中文分词在中文文本处理中非常重要，因为汉字并没有空格或其他分隔符来区分单词，这给中文文本处理带来了困难。', 10.517306327819824)
   ('本文将介绍中文分词的原理和常用方法。', 19.19382667541504)
   ('中文分词的原理是基于词语在语料库中的频率来确定一个句子中的分词方案。', 18.930423736572266)
   ('分词的目标是将句子中的每个字符分配到一个词语中，从而获得有意义的词语序列。', 13.634822845458984)
   ('这样，文本就可以被更好地理解和分析。', 10.56257438659668)
   ('中文分词的方法有很多，其中最常用的是基于统计学的方法和基于规则的方法。', 8.299059867858887)
   ('基于统计学的方法通过对大量中文文本语料库的分析来得出一个概率模型，该模型根据词频和词语之间的关联性来判断每个字符的最佳分词位置。', 12.321337699890137)
   ('这种方法的优点是可以自适应地适应不同类型的文本，并且能够处理新词和生僻词，但是也有一些缺点，例如需要大量的语料库来训练模型，模型的效果受到数据质量和模型参数的影响。', 10.90810489654541)
   ('基于规则的方法则是依赖于一系列语法规则和词库，例如词典、词性标注、句法分析等，来判断每个字符的最佳分词位置。', 17.036846160888672)
   ('这种方法的优点是可以精确地控制分词结果，适合处理特定领域的文本，但是需要手动构建规则和词库，工作量比较大。', 14.91514778137207)
   ('除了基于统计学的方法和基于规则的方法，还有一些混合方法，例如将基于统计学的方法和基于规则的方法结合起来，以获得更好的分词效果。', 5.419482707977295)
   ('总之，中文分词是中文文本处理中的重要技术之一。', 15.306158065795898)
   ('在实际应用中，根据不同的需求和场景，可以选择不同的分词方法来获得最佳的分词结果。', 7.240055084228516)
   Average Perplexity Score:  12.95176352773394
   Burstiness Score:  4.184883485157172
