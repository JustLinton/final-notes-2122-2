#### 引言

###### 分时系统中，100个进程，为保证响应时间不超过2s，时间片最大长度？

```
需要在2s内，把100个进程都遍历到了（虽然应该是不能结束），但是只要遍历到了就是响应了。所以是20ms。
```

###### 画出甘特图，计算CPU利用率？

```
从甘特图中找出CPU空闲的时间段，用总时间减去这些时间段除以总时间，就是CPU利用率。
CPU利用率即CPU忙时间除以总时间。
```

###### OS对进程的管理和控制使用是通过`原语`实现的。

###### 进程等待CPU时，`不处于阻塞态`，虽然他进入了等待。

###### 若系统中没有运行进程，那么当前一定`没有就绪进程`。

###### 多线程和多任务的区别？

- 多任务对操作系统而言的。该操作系统可以并发执行多个任务（支持多道程序）。
