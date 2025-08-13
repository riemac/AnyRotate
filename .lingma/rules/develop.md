---
trigger: model_decision
description: IsaacLab开发规则
---

## IsaacLab开发规则

当在开发过程中遇到以下情况时，应按照以下规则处理：

### 1. IsaacLab API或概念不清楚时的处理流程

1. **优先使用context7工具查询**
   - 当遇到IsaacLab相关的API、概念不清楚时，应首先使用`mcp_context7_resolve-library-id`和`mcp_context7_get-library-docs`工具查询官方文档
   - 通过这种方式可以获取最准确、最新的API信息，避免产生幻觉或使用不存在的API

2. **其次使用网络搜索**
   - 如果context7工具未能解决疑问，可以使用网络搜索工具进一步查找相关信息
   - 搜索时应优先考虑官方文档、GitHub仓库和权威技术博客

3. **避免产生幻觉**
   - 不要凭空编造API或功能
   - 不要假设存在未验证的函数或类
   - 始终以查询到的官方文档为准

## 开发实践建议

- 在编写代码前，先查询相关模块的文档和示例代码
- 参考IsaacLab官方提供的环境配置和实现方式
- 保持代码风格与IsaacLab项目一致
- 充分利用IsaacLab提供的工具和框架功能