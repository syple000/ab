#ifndef CONFIG_CONFIG_H
#define CONFIG_CONFIG_H

#define ENABLE_CUDA false
#define ENABLE_TENSOR_EXCEPTION true // 如果禁止，问题会在其它步骤导致异常，可从错误日志观测
#define ENABLE_GRAD_DESCENT_ECHO_GRAD true
#define ENABLE_GRAD_DESCENNT_RAND_STEP true // 梯度下降中随机大步长，跳出局部最优
#define ENABLE_GRAD_DESCENNT_RAND_STEP_RATIO 0.1

#endif