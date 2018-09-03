# PyTorch实战指南 第六章  Dog.VS.Cat
这是根据深度学习框架：PyTorch入门与实践这本书的第六章写的代码，是关于猫狗识别的，在这个过程中，一边看，一边写，刚开始是运行作者已经写好的代码，后来自己在jupyter上进行复制的复现，发现import无法导入ipynb文件，在使用了Ipynb_importer.py之后可以实现同一文件内导入ipynb模块，如果是在其他文件中进行导入，会有点费事，以下会记录Ipynb_importer.py的用法。因为费事，自己开始开始使用pycharm+jupyter的方式，直接自己根据作者提供的源码进行编写，在编写的过程中接受作者的思想。用pycharm的不方便的地方是无法直接运行测试，所以采取的是对自己不熟悉的模块或者方法，用jupyter进行测试，而直接编写则是pycharm。但是感觉pycharm还是没有那么好用，可能是自己用的少。我是按照data、model、util、main+config、requirement的顺序编写的。在编写函数的过程中，因为刚开始不理解各个模块是怎么组织起来的，所以都是从简单的开始，所以函数的位置和作者的不一样，其中对于model.save和model.load、vis.plot和vis.log的封装让我感觉很有意思，刚开始是编写的时候只能直接打上问号，因为不懂这么编写的意义，但在编写主函数main的时候才感觉到了这种编写的好处，基本把模型训练和对模型、结果的处理完全分离开，避免了耦合性很强的后果。
____
# Ipynb_importer.py
我通过几次测试发现，import Ipynb_importer 只需要放在你的当前要运行的文件中即可，然后在其他文件下的__init__.py 中导入所有的当前文件夹中的Module，就像这样    
/first/second/models/    
-----__init__.py    
------------- None
-----BasicModule.ipynb    
-----AlexNet.ipynb    
----------from models.BasicModule import BasicModule

/first/main.py    
import Ipynb_importer
from models import AlexNet    

之所以在AlexNet中写models.BasicModule是因为直接导入BasicModule会报错，我根据__dict__的输出发现有问题，这一点和[官网](https://jupyter-notebook.readthedocs.io/en/latest/examples/Notebook/Importing%20Notebooks.html)介绍的有一点区别，我没有实现官网说明的跨文件夹导入。因为如果改文件夹导入的话，models.BasicModule要接着换成相应的名字，与我预想的不一致，我预想的是不管在哪里导入，已经导入的应该不受影响才对。
____
# ipynb-py.sh
之后发现了这个神器，可以把ipynb转化成.py，还是挺好用的，转化之后也没问题。
___
同时，借助这次实验，自己对python的掌握也更深了一点。
_____
不过对于网络的构成还是有一些问题，那就是网络为什么这么写，这应该属于理论的东西。还需要进一步加强。    
_____

这次实验一共用了三天才完全搞懂，可以说其中涉及到的函数的用法基本都明白了。     
本意是记录自己，不过如果有任何问题，欢迎交流。     
____

