#### 3-存储程序

- 程序一旦运行，不需要人工干预即可自动执行

#### 3-冯诺依曼机器特点

- 采用存储程序的方式
- 计算机由5大部分构成
- 数据和指令在存储器中，以二进制表示，且没有区别，但计算机能区分
  - 指令由操作码和地址码构成
- 控制器能自动执行指令，运算器能进行算术和逻辑运算
- 操作人员使用IO操作计算机

#### 4-通用寄存器组（GPRs）

- General Propose Registers.用于临时存放CPU从主存取来的数据或运算中间结果

#### 6-时钟周期

- 时钟信号的宽度称为时钟周期

#### 7-程序语言和自然语言的区别

- 程序语言不存在二义性，且具有严格的执行顺序

#### 7-机器语言

- 即0-1序列
- 汇编语言必须转换成机器语言才能执行

#### 8-翻译程序

- 翻译程序分为汇编程序assembler、解释程序interpreter、编译程序compiler三个大类。
- 汇编程序把汇编语言转换为机器语言
- 解释程序把源程序**按执行顺序**翻译成机器指令并**立即执行**
- 编译程序把**高级语言**源程序翻译成汇编或机器语言

#### 8-源程序和目标程序，源语言和目标语言

- 前者是源代码写的程序，后者是翻译生成的程序。“语言”即所使用的语言。

#### 9-mips指令集

- mips是一种risc指令集。即microcomputer without interlocked pipeline stages.

#### 10-gcc编译过程

`hello.c`为例

- （高级语言源程序）预处理：对“#”定义的宏进行预先处理，包括把头文件嵌入等操作，生成`hello.i`。它仍是高级语言源程序。
- （汇编语言源程序）编译：对`hello.i`进行编译，生成汇编语言源程序`hello.s`。
- （可重定位目标程序）汇编：对`hello.s`进行汇编，生成机器语言的可重定位目标程序（relocatable object file） `hello.o`
- （可执行目标程序）链接：把`hello.o`和其他文件例如`printf.o`进行合并，生成可执行文件`hello.out`
  - 可执行文件又称可执行目标程序。可执行目标程序和可重定位目标程序的区别是前者是可执行的，而后者需要经过链接。

#### 11-读入键盘字符、输出字符到屏幕

- 键盘字符送到了CPU的寄存器中，随后送到主存中（输入缓冲区）
- 数据在送上总线之前，都要**先缓存**在存储部件（不是存储器）中。
  - 因此，在端口中，一般都有数据缓冲。

#### 12-南桥北桥

- 北桥芯片负责处理高速部件的通信，例如RAM
- 南桥芯片负责处理IO总线通信，例如外部设备、硬盘等

#### 12-端口

- 数据缓冲寄存器、命令字寄存器、状态字寄存器
- 端口可以进行独立编址，也可进行存储器映射编址。前者是自己一套IO地址空间，后者是把内存地址的一部分拿出来给IO编址。

#### 13-操作系统

- **操作系统是对计算机底层结构和硬件的一种抽象**

#### 13-计算机体系结构（指令集体系结构，ISA）

- Instruction Set Architecture
- 即计算机硬件和软件之间的交界面，软硬件接口
- 是软件对计算机硬件的感知方式
- ISA规定了一套指令集，即计算机机器语言的一套**设计规范**。例如Intel x86 ISA下有很多款CPU，但是由于使用了相同的ISA，所以在i9-9900k上跑的程序在上也能在AMD 3900X上跑。
- **ISA是整个计算机系统的核心。**ISA集中体现了硬件的特性，软件在ISA上跑。

#### 14-微体系架构（微架构）

- 一套ISA的**具体实现**就是这个ISA的微体系架构。
- 例如加法器可以使用串行进位，也可使用超前进位。

#### 14-高级语言`翻译`程序的前端和后端

- 高级语言程序的编译可以理解为语法分析、语义分析、中间代码生成、中间代码优化、目标代码生成、目标代码优化。
- 把中间代码生成**及其**之前的操作称为前端，之后的操作称为后端。
- 可以理解为：前端是检查ASCII写的程序并转化为汇编，后端是优化汇编并汇编成为目标程序。

#### 14-未定义行为和未确定行为

- undefined behavior：例如C90规定：在printf中使用错误的类型描述符就是未定义行为。它会导致每次执行的结果都不一样，或者在不同的微架构下执行结果都不一样。（即玄学行为）
  - 未定义行为即官方明确表示“我们不保证这么写他能跑”。
  - 原因：gcc明确给出了警告：这么写，**我们也不保证**他能跑出什么结果来。
- unspecified behavior：例如C对char到底是unsigned还是signed没有作出要求，所以在不同微架构下的gcc可能会对这个char给出不一样的解释，导致在不同微架构下，其执行结果不同。
  - 未确定行为即未明确给出规定的，模棱两可的。
  - 原因：gcc**没有强制规定**char属于什么类型，于是不同gcc的实现对他的处理有一定差异。

#### 15-ABI接口和API接口

- ABI接口可以类比API接口，但前者的层次非常低，包含了和硬件相关的接口，但是后者只是高层次的库之间的接口而已。
- ABI接口是在给定的ISA、给定的OS下，规定的及其级别的目标代码层接口。
  - ABI规定了诸如OS的系统调用、可执行文件格式、寄存器使用约定、内存地址划分等规范。因此，它不仅取决于ISA，还取决于OS。
  - 可以想象，如果在某ABI下编译好的程序，拿到另一台有相同ABI的电脑上跑，是可以正常跑的。

#### 16-计算机软件分类

- 系统软件
  - 包括OS、DBMS、Compiler、实用程序（例如KDE Plasma）
  - OS对硬件进行管理监视，对软件提供环境，对用户提供界面
  - 语言处理程序负责提供编程环境、装入运行环境等

#### 16-计算机用户分类

- 最终用户、系统管理员、应用程序员、系统程序员

#### 17-什么是透明

- 一种客观存在的东西，在某个**角度来看好像**不存在一样

#### 17-几种透明虚拟机

- 机器语言程序员看到的机器是ISA以上的部分，以下的部分透明了，称为机器语言虚拟机。
- 系统程序员看到的机器是装备了OS的机器，看到的是OS以上的部分，以下的部分透明了，称为操作系统虚拟机。
- 以此类推，还与汇编语言虚拟机、高级语言虚拟机。

#### 18-衡量计算机系统性能的定义

- 吞吐率：单位时间完成的工作数量
- 响应时间：从作业提交到给出相应（或作业完成）经过的时间
- CPI：一条指令执行所需的时钟周期数，一般采用`平均CPI`。对于一段给定的程序，可以求**针对**这段程序的平均CPI，即该程序的`综合CPI`（程序总指令数除以程序花费的总时钟周期数）。

#### 19-用户视角下的CPU时间（如何衡量计算机的好坏）

- 分为系统CPU时间（操作系统各操作占用的时间）、用户CPU时间（用户程序所占用的时间，**是用户体验的部分**）、其他CPU时间（例如等待IO时间、其他用户查程序占用的时间）。
- **衡量计算机性能好坏，往往通过用户CPU时间来衡量**。计算机的性能可以看做是用户CPU时间的**倒数**，也就是执行这一个用户程序需要多少用户CPU时间，该时间越短说明程序执行速度越快。
- 用户CPU时间=`CPI*这段程序的指令数*时钟周期宽度`
  - 可见，计算机的速度由CPI、指令条数、主频共同决定。他们互相制约，共同影响计算机的性能。

#### 21-MIPS法、Gibson混合法、FLOPS法

- Gibson混合法（对一条指令所需执行时间进行加权平均）：第i种指令的占比是wi，其所用时间是ti，则sigma(wi*ti)就是计算机**一条指令**的平均执行时间。
  - 如果时间单位是节拍，那么这就是CPI。
  - 如果时间单位是s，那么对他取倒数就是MIPS。
- 峰值MIPS，相对MIPS（P21）

#### 22-benchmark程序（基准程序）

- 基准程序是一种小程序，在不同电脑上运行，比较其执行时间，以对比出电脑性能的差异。
- 不排除厂商对基准程序中的瓶颈指令编写特定的编译器进行优化，以得出离谱的跑分的情况。（根据阿姆达尔 Amdahl 定律）。

#### 23-阿姆达尔定律 Amdahl's Law

- 对计算机系统的硬件或软件部分进行改进带来的性能提升程度**取决于该部分被使用的频率或占总执行时间的比例**。
- 定律的两种公式（P23）和例题
- 极限加速比：如果仅对某关键部分进行改进，而不去考虑其他部分，那么这种改进带来的性能提升是有上限的，例如P24例子，你怎么改进都不可能让性能提升5倍。
- 如果仅对使用较少的部分进行改进，带来的性能提升可能趋近于0.

#### 30-离散化

- 必须是数字化的信息，计算机才能处理，所以输入设备都有着“**离散化，然后编码**”的功能。

#### 31-计算机为什么采用二进制

- 找到能稳定保持两种状态的物理器件比较简单。
- （便于算术运算）二进制的运算比较简单。
- （便于逻辑运算）二进制对应真、假。便与实现数字逻辑。

#### 31-机器指令所能处理的数据类型

- 分为数值型和非数值型数据。
  - 非数值型包括编码字符、逻辑数据
  - 数值型包括二进制数、二进制编码的十进制数。
    - 二进制数分为整数（有符号定点数、无符号定点数）和实数（浮点数）。

#### 35-进制转换

- 进制转换的原则是：整数部分和小数部分分别来做。
- R进制转10进制，以8转10为例：
  - 整数部分：按权展开。注意个位是8的0次方。
  - 小数部分：按权展开。注意第一个小数位是8的-1次方。
- 10进制转R进制，以10转8为例：
  - 整数部分：每次除以8，拿到余数作为从右往左罗列的新位，然后继续对商进行上述操作，直到商0.
  - 小数部分：每次乘以8，拿到整数部分作为小数点后从左向右罗列的新位，然后继续对小数部分进行上述操作，直到没有小数部分。
    - 如果永远都有小数部分，那么就近似处理。例如(0.63)D=(0.1010...)B。

#### 36-定点数和浮点数

- 计算机无法表示小数点。因此要表示小数，只能通过约定小数点所在位置的方式实现。我们约定整数的小数点位置在最低位之后，定点小数的小数点在第一位后，浮点小数的小数点可浮动。

- 定点整数和定点小数统称为“定点数”。

- 浮点数三要素：尾数（原码表示），基数，指数（移码表示）

  - 指数的表示范围直接决定了该浮点数的表示范围。**指数的值决定了小数点的位置**。

  - 尾数的表示范围决定了浮点数的精度。尾数越长，代表其有效位越多，也就是精度越大。

  - $$
    2^{-n}\times2^{-(2^m-1)}\le|M|\le(1-2^{-n})\times2^{2^m-1}
    $$

    设尾数**小数点后**n位，指数**除了符号位**有m位，则该浮点数的绝对值的范围是.

    - 对于尾数，其是有符号定点小数，所以绝对值最小的数就是其小数点后最后一位是1，因为小数点后1位是2^-1,所以第n位就是2^-n。相反，如果要最大的这种小数，直接用1减，就能得到小数点后全1，显然最大。
      - 尾数我们要求绝对值最大和最小。因为上面那个式子我们看的是绝对值。
    - 对于指数，其是有符号定点整数，所以最小数就是负的全1，最大数就是正的全1.
      - 指数我们要求带符号的最大和最小。因为其符号不影响浮点数的符号。

  - 如果要拼凑绝对值最小的浮点数，只需要尾数绝对值最小，且指数带符号最小；拼最大的则反过来。

  - 可以看到，由于指数的加入，整个浮点数的表示范围远超了它带的这个定点小数的范围。

  - 区分“范围”和“精度”：范围是数字本身的大小，精度是有多少位有效数字。**即一个是数值的大小，一个是有效位的多少。**

#### 37-机器数和真值

- 机器数即数字在机器内的编码表示，例如原、反、补、移。真值是人对数字的表示
- 原码表示的问题是0的表示不唯一，且运算不便（需要判断正负）
- 补码的新理解
  - 对于整数，补码就是他本身；对于负数，补码=模减去该负数的绝对值（实际上就是对该负数取模）（**但是注意，这里的运算是以无符号数的方式进行，也就是最高位也参与减法**）。
  - 假设当前包括符号位是n位，模的就是2的n次方（于是就可以做到这个模实际上是1后面n个0.），那么让待表示的负数取这个模可想而知，实际上就是让这个模减去这个负数的绝对值（注意是无符号数的运算方法。）
  - 于是就得到了奇妙的性质：
    - 对于一个补码表示的数，如果它最高位是1，那么它一定是负数，因为只有负数才是通过与模做差得来的，而做差会导致最高位1的出现。
    - 如果把补码表示的这个数字整体看作无符号数，那么随着它的增大，它的真值也是单调递增的（除了经过0的时候是一个跳跃，即从正数大的数跳到负数最小的（绝对值最大）那个数）。因为对于负数来说是做减法，减法以后剩下的1越多，代表原来1越少，也就是绝对值越小，也就是真值越大。
    - 即：0->1->+max->-max->-1.（-max是指负数里面的绝对值最大的）
  - 同一真值，编码位数不同，模不同，从而导致表示出来的补码不同。（例2.12）
  - 因为$$2^{n-1}$$ 的补码是100....0，是个负数，且-2^(n-1)的补码也是100...0（因为一个是2^n+2^(n-1），一个是2^n-2^(n-1))，所以他俩就冲突了，而且显然对于正的那个2^(n-1)来说不合适。所以补码的表示范围的那个等于号最终还是花落-2^(n-1)家。
  - 因为0的补码是2^n+0=2^n-0=0，所以不存在+0和-0的区别了，因此100...0给到-2^(n-1)那里也是`正确`的。
- 移码
  - 浮点数的指数往往需要对阶操作，所以需要一种表示方法，能在表示范围上完全单调递增（补码只能保证符号内单增）
  - 只需要给数加上一个偏移量（bias），这个偏移量一般取 $2^{n-1}$，也就是`模右移一位`。这样一来，就能在表示范围内实现单增了。
    - 注意，n是整个移码的长度，所以有效位一共有n-1位，而 $2^{n-1}$为1的那一位是最高位（不属于有效位），所以能达到这种效果。
    - 移码中，最高位也不是有效位，它类似于符号位，与符号有关。
  - 数的排列是这样的：-max->-1->-0->+0->1->+max.（-max是指负数里面的绝对值最大的）

#### 43-C中有符号数和无符号数的转换

- 在C中，这两种数之间可以进行强制转换，但是转换前后`机器数不变`，而只是`解释`这个机器数的方法变了。
- 因此一个很小的负数如果转换成无符号数，会变成一个很大的正数。
- 当无符号数和有符号数进行运算，有符号数会被强制换为无符号数，此时会如上一条所说，产生unexpected现象。例如例2.21

#### 44-C中数值常量类型的确定

- 对于一个常量数字，在C90下，其分水岭是 $2^{31}-1$、 $2^{32}-1$、 $2^{63}-1$、 $2^{64}-1$。分别对应int，unsigned int，long long，unsigned long long。因此如例2.21所述，如果一个数字是 $2^{32}$，那么它被解释成了unsigned int。这个时候如果再去和其他int做运算，就会出现问题。
- 在C99下，取消了$2^{32}-1$这个分水岭。也就是取消了unsigned int这个判断，其空出来这块区域直接由long long代替。

#### 45-阶码造成的上溢、下溢、机器零

- `阶码`没法再小了，导致比较小的数无法表示，这种称为下溢出。
- 下溢可用机器零处理。因为当尾数=0，阶码变得没有意义。所以只要尾数是0，整个数就是0了，称为机器0.因为阶码没什么用，所以机器0的表示显然不唯一。
- 上溢是真的溢出了。`阶码`超出了可表示的最大的绝对值范围。
- **上、下溢出一般是由阶码引起的，而与尾数无关。因为阶码负责数字的表示范围。**

#### 45-尾数的规格化

- 左规：尾数有效位前的0过多，影响精度，需要左规。
- 右规：尾数的有效位跑到小数点左边了，不符合表示格式，需要右规。

#### 46-IEEE 754标准

- 目前几乎所有的计算机都使用IEEE 754标准。该标准规定了32位单精度浮点数、64位双精度浮点数类型。
- 尾数采用带有一个隐藏位的`原码`表示，阶码采用偏置-1的`移码`表示。
  - 因此，只需要一个符号位，用于表示整个浮点数的符号。这个符号位实际上也是尾数的符号位，因为阶码用移码表示了，不需要符号位了。
  - 于是得出754的格式：符号|阶码|尾数。
- 尾数隐藏位
  - 因为规格化以后尾数都是1，所以这个1就不必写了，所以例如32位的类型中，尾数有23位，但是实际上可以表示24位内容。`扩大了精度。`
  - 可以把隐藏位看做是在小数点之前。
- 偏置-1
  - 和一般的移码不一样，例如有n位，那我偏置就是 $2^{n-1}$，但是这里偏置是$2^{n-1}-1$。`这样扩大了表示范围。`
  - 即全0就是最小负数。

#### 46-IEEE 754的特殊值

- 特殊值：阶码全0或全1.
- 正常数字：阶码非全0，非全1.

| 阶码         | 尾数 | 含义                           |
| ------------ | ---- | ------------------------------ |
| 全0          | 全0  | +0或-0（符号取决于符号位）     |
| 全0          | 非0  | 非规格化数                     |
| 全1          | 全0  | +inf或-inf（符号取决于符号位） |
| 全1          | 非0  | NaN（尾数可携带错误信息）      |
| 非全0，非全1 | 任意 | 正常数字                       |

#### 47-IEEE 754的非规格化数及其逐级下溢

- **如果尾数隐藏位是0，且阶码是最小阶码（-126或-1022）**，那么这是一个IEEE 754非规格化数。
- 逐级下溢是利用非规格化数实现的，用于保证当运算过程中，阶码超出了表示下界时，程序仍然能正常运行。
  - 以32位（单精度）浮点数为例，其阶码的最小值是-126.
    - 得出机器零的情况（发生下溢出的情况）：例如 $2\times10^{-90}\times2\times10^{-90} = 1.04\times10^{-180} = 0.104\times10^{-179}=...=0.0$，即每次都右移动，当把有效位都移没了，但是仍然没有把阶码从下溢出的范围内移出来，就直接变成了机器零。
    - 避免下溢出的情况：例如 $2\times10^{-90}\times2\times10^{-37} = 1.04\times10^{-127} = 0.104\times10^{-126}$，此时通过一定次数的右移，把阶码从下溢出的范围内移出来了，顺利完成了运算，且显然得到了规格化的数。
- 非规格化数和规格化数是没有冲突的。非规格化数的加入，利用了一块本来没法利用的数字区间。也就是区间 $[0.0...0\times2^{-126},0.0...1\times2^{-126}]$，因为规格化数的第一个有效位必须是1，所以显然这个区间被浪费掉了。而根据规格化数的定义，这里就给完美地用上了。

#### 47-IEEE 754精度和阶码的关系

- 类比：$[00000001\times10^2，99999999\times10^2]$所表示的区间长度要比$[00000001\times10^9，99999999\times10^9]$小得多，同时，有效数之间的间距也要小得多。也就是前者更加精确，但是范围小；后者范围大，但是丧失了精确度。
- 对比 $[0.1...0\times2^{-126},0.1...1\times2^{-126}]$和 $[0.1...0\times2^{-125},0.1...1\times2^{-125}]$，也是一个道理。
- 而且你会发现，非规格化数由于在 $[0.0...0\times2^{-126},0.0...1\times2^{-126}]$，所以它和区间 $[0.1...0\times2^{-126},0.1...1\times2^{-126}]$的长度、有效数字间隔都是一样的，就好像是复制过来了。

#### 48-IEEE 754的NaN

- 最高有效位是1时，是不发信号的NaN，不会产生错误信息；否则是发信号的NaN，会产生错误信息。
- 其余的位可以携带异常。
- 当进行无数学解释的运算，会产生NaN。**基本上都是那些类似洛必达法则的关于0和无穷的运算。**

#### 50-C中的强制类型转换

- 一般情况下int是32位，float是32位，double是64位。
- int转float会舍去一些有效位，因为float有效位不足32个。
- int、float转double没有问题。double有52位尾数，也是比32位的int要多的。
- double转float时，可能会溢出（对`阶码`）。且精确度降低（对`尾数`）。
- float、double转int时，由于int没有小数部分，所以会`向0方向`截断。
- 注意，在char转int时，可能出现问题。因为char是unsigned还是signed是未给出要求的，所以这种转换是`未确定行为`。例如0xff转换为int，x86认为这是一个signed char，由于采用补码表示，其在高位补0，也就是补1，所以最终得到了0xffffffff这个int。但是如果在RISC-V上跑，他认为是unsigned char，那么高位只需要补0，则得到了0x000000ff这个int。

#### 50-C中的浮点数运算

- 例如 $1.79\times10^{308}+1.0\times10^1$，那么就会出现问题，因为后者要向前者对阶（`小对大`），其有效位在右移的过程中会丢掉那唯一的1，导致变成0.于是运算就变成了 $1.79\times10^{308}+0$.

#### 51-有权码和无权码

- 以BCD为例，有权BCD一般是8421码，无权BCD一般是余3码、格雷码。

#### 52-西文字符和ASCII

- 字符集：各种字母、符号构成的集合。
- 码表：字符集每个字符和编码对应构成的表。
- ASCII的规律
  - 用7位二进制数来编码（因为早期8位机下，可以留一位给校验位，如果需要的话）
  - 数字的高3位是011，低4位正好是其8421BCD码。
  - 字母的第5位如果是0，则是大写，否则是小写。这给大小写转换带来了便利。

#### 54-汉字输入码（外码），国标码（国标交换码），机内码（汉字内码）

- 输入码即用键盘按键编码构成的序列，来确定一个汉字
- 国标码是区位码+32.（由于信息传输限制）
  - GB2312属于国标码
  - 类似的编码还有CJK、微软的Unicode
- 机内码是把国标码的两个字节的最高位都设为1.（由于避免与ASCII撞车）
  - 机内码是汉字编码在计算机内的实际存在形式

#### 55-汉字的描述

- 点阵描述——位图思想
- 轮廓描述——矢量图思想

#### 55-字长和字

- 字长即CPU用于整数运算的数据通路的长度。
- 字可以不等于字长。**例如现在的64位机，其字仍然是16位，但是字长却是64位。**
  - 因此，即便是在64位机中，我们仍然定义32位的数据类型为“双字”。
  - 字是信息处理的单位，只是用来规定数据类型的长度。
  - 字长是计算机处理能力的反映。**例如64位机可以同时处理4个字，可以看出性能比较强大。**
  - **因此，可以认为字节永远都是8bits，字永远都等于2个字节。**

#### 56-C中数据类型的宽度

| C类型  | 32位机(单位：B) | 64位机(单位：B) |
| ------ | --------------- | --------------- |
| char   | 1               | 1               |
| short  | 2               | 2               |
| int    | 4               | 4               |
| long   | 4               | 8               |
| char * | 4               | 8               |
| float  | 4               | 4               |
| double | 8               | 8               |

- 可见，就算是同种数据类型，在不同的ABI下，宽度不一定相同。因此需要取决于**具体的ABI。**

#### 57-LSB和MSB

- 即最低（高）有效位（字节）least/most significant bit/byte
  - 注意，有效最高、低位是指考虑了这段数据的含义在内后，给出的定义。例如一个数据如果倒着放，那么低地址实际上对应了MSB，这种情况下就是大端存储。
- 产生原因：仅仅说“最左边的位”、“最右边的位”会产生歧义。因为数据可以小端、大端存储。
- 使用字节是因为现代计算机一般**按字节**编址。

#### 57-字节排序问题——大小端存储

> **大端存储是顺着放。**
>
> （因为MSB是数据最高位，假设我们大端，那么最高位就是在较小的地址处，那么符合我们日常书写的习惯，即从左向右）

- 计算机按字节编址，但是数据的长度往往不止一个字节，例如int是4个字节，这4个字节是**正着放**到连续内存中，还是**倒着放**在连续内存中，即字节排序问题。
- 大端存储，即数据的地址是MSB所在地址；小端存储，即数据的地址是LSB所在地址。（我们认为数据地址是这几个字节的起始地址。）
- x86架构采用的是小端方式，mips采用大端方式。
- 大小端存储问题**同样存在于软件领域**，例如gif格式是小端，jpeg则是大端。
- 采用不同字节排序方式的计算机（软件）之间不能进行直接通信。需要事先转换。

#### 61-逻辑运算和按位运算

- 逻辑运算的结果只有true和false；按位运算是一种数值的运算，是把操作数的各个二进制位分别进行逻辑运算从而得到一个新的二进制数，结果是新的二进制数。

#### 61-C中的移位运算

- 往往对于无符号数进行逻辑移位，对于有符号数进行算术移位。
- 技巧：因为左移相当于乘2，所以有溢出的可能。只要移出去了1，就是溢出了；因为右移相当于除以2，所以涉及到不能整除的问题。只要移出去了1，就是不能正处2（因为最低位是1必然是奇数）。

#### 62-C中的扩展与截断

- C没有扩展、截断运算符，但是它在强制类型转换时会自动进行。
- 扩展分为0扩展（高位全补0，用于无符号数）、符号扩展（高位全都用符号位填充，用于补码表示的有符号数。说成填充符号位没毛病，因为如果是负数，补1相当于补0，正数就不用说了。）
- 截断发生在长数强制转换短数时，把高位删掉。例如把32768（`0x00008000`）这个int截断成short，则删去高2个字节，得到`0x8000`，即`1000 0000 0000 0000`，是-32768的补码（==这不是那个负0空出来的吗==）。
- 截断属于`未定义行为`，C语言不保证能输出合理的结果。因为截断非常容易溢出，长数的表示范围远大于短数。
  - 可以看出，特别是对于有符号数，只要你长数的有效位数大于短数总位数，只要截断必然出来个负数。

#### 64-OF和ZF标志位

- 如果结果所有位都是0，则ZF。
- **如果两个加数的符号位相等，且结果符号位和他们不相等，则OF。**
