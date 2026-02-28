import streamlit as st
import pandas as pd
from pdfminer.high_level import extract_text
from openai import OpenAI
import json
import re

# 1. 页面配置与标题
st.set_page_config(page_title="AI简历助手-厦门物流专用", layout="wide")
st.title("AI简历助手-厦门物流专用")

# 2. 侧边栏：配置密钥与岗位选择
with st.sidebar:
    st.header("系统设置")
    api_key = st.text_input("请输入 API Key (DeepSeek/OpenAI)", type="password")
    base_url = st.text_input("API 接口地址", value="https://api.deepseek.com")
    model_name = st.text_input("模型名称", value="deepseek-chat")
    
    st.markdown("---")
    st.header("评分设置")
    job_type = st.selectbox(
        "请选择要评估的岗位类型", 
        [
            "1. 国际物流业务销售、集装箱货代销售、货代业务员 (社招)", 
            "2. 仓储物流销售、管理总经理 (社招)", 
            "3. 党建工作/文书岗位 (社招)", 
            "4. 应届毕业生"
        ]
    )

# 3. 根据不同的岗位，加载对应的评分标准
if job_type == "1. 国际物流业务销售、集装箱货代销售、货代业务员 (社招)":
    SCORING_PROMPT = """
    你是一名资深HR。请根据以下标准对简历打分（满分100分）：
    1. 求职状态 (10分)：离职随时到岗 (10) / 在职看机会 (5) / 暂不急跳槽 (0)
    2. 相关行业经验 (40分)：3-5年匹配同岗位的经验 (40) / 3-5年其它行业销售经验 (35) / 3-5年行业内工作经验 (25) 
    3. 企业背景 (35分)：世界500强/知名大货代 (35) / 中型规范企业 (25) / 行业相关普通公司 (20)
    4. 学历排序 (15分)：985或211或双一流或海事类名校(集美、上海海事、大连海事等) (15) / 普通本科院校 (10) / 专科 (5)
    请必须输出JSON格式：{"姓名": "", "总分": 0, "维度得分": {"状态":0, "经验":0, "背景":0, "学历":0}, "简评": "此处给出扣分原因或亮点简评"}
    """

elif job_type == "2. 仓储物流销售、管理总经理 (社招)":
    SCORING_PROMPT = """
    你是一名资深HR。请根据以下标准对简历打分（满分100分）：
    1. 求职状态 (10分)：离职随时到岗 (10) / 在职看机会 (5) / 暂不急跳槽 (0)
    2. 相关行业经验 (40分)：3-7年仓储相关总经理经验 (40) / 3-7年仓储相关销售或管理岗位经验 (35) / 3-7年行业内其他管理岗位经验 (30)
    3. 企业背景 (35分)：世界500强/知名大货代 (35) / 中型规范企业 (25) / 行业相关普通公司 (20)
    4. 学历排序 (15分)：985或211或双一流或海事类名校(集美、上海海事、大连海事等) (15) / 普通本科院校 (10) / 专科 (5)
    请必须输出JSON格式：{"姓名": "", "总分": 0, "维度得分": {"状态":0, "经验":0, "背景":0, "学历":0}, "简评": "此处给出扣分原因或亮点简评"}
    """

elif job_type == "3. 党建工作/文书岗位 (社招)":
    SCORING_PROMPT = """
    你是一名资深HR。请根据以下标准对简历打分（满分100分）：
    1. 求职状态 (10分)：离职随时到岗 (10) / 在职看机会 (5) / 暂不急跳槽 (0)
    2. 相关行业经验 (30分)：3-5年党建岗位的经验 (30) / 3-5年其它文书岗位工作经验 (20)
    3. 企业背景 (25分)：世界500强/知名大货代 (25) / 中型规范企业 (15) / 行业相关普通公司 (10)
    4. 政治面貌 (25分)：中共党员 (25) / 预备党员 (10) / 其他 (0)
    5. 学历排序 (10分)：双一流以上院校及或行业内名校（集美大学、上海海事大学、大连海事大学）(10) / 普通本科院校 (5) / 专科以下 (0)
    请必须输出JSON格式：{"姓名": "", "总分": 0, "维度得分": {"状态":0, "经验":0, "背景":0, "政治面貌":0, "学历":0}, "简评": "此处给出扣分原因或亮点简评"}
    """

else:
    # 最新的 "应届毕业生" 标准
    SCORING_PROMPT = """
    你是一名资深HR。请根据以下【应届生】标准打分（满分100分）：
    1. 学历排序 (40分)：双一流、985、211及重点海事类院校（上海海事大学、集美大学）、境外全球QS排名前100或QS专业排名前50院校及部分一线生产经营单位所在行业对应的头部院校应届毕业生(40) / 普通本科院校(30) / 其他 (10)。
    2. 专业排序 (30分)：物流管理、物流工程、交通运输、交通工程、国际经济与贸易、计算机科学与技术、物流工程与管理等理工科类根据相关度得分。满分30分。
    3. 英语水平或特定岗位证书需求 (10分)：英语水平 雅思 6.5,托福 90,专八、 CET6(10) / CET4(6) / 无 (0)。
    4. 实习经历 (10分)：有货代/物流相关实习(10) / 有其他行业实习(5) / 无(0)。
    5. 综合素质 (10分)：有学生会/社团干部经历或获奖(10) / 普通获奖经历(5) / 无(0)。
    请必须输出JSON格式：{"姓名": "", "总分": 0, "维度得分": {"学历":0, "专业":0, "英语及证书":0, "实习":0, "综合素质":0}, "简评": "此处给出评估依据"}
    """

# 4. 上传文件与处理逻辑
uploaded_files = st.file_uploader("请上传求职者 PDF 简历（支持多选）", type="pdf", accept_multiple_files=True)

if st.button("开始 AI 筛选排名") and uploaded_files:
    if not api_key:
        st.error("请先在左侧输入 API Key！")
    else:
        client = OpenAI(api_key=api_key, base_url=base_url)
        all_results = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            # 提取文件名及里面的数字编号
            filename = file.name
            match = re.search(r'\d+', filename)
            resume_id = match.group() if match else "无编号" 
            
            # 提取文本
            text = extract_text(file)
            
            # 调用 AI
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": f"{SCORING_PROMPT}\n\n简历文本如下：\n{text[:3500]}"}],
                    response_format={'type': 'json_object'}
                )
                res_json = json.loads(response.choices[0].message.content)
                
                # 组合数据，加上简历编号
                res_json["简历编号"] = resume_id
                res_json["原文件名"] = filename
                
                all_results.append(res_json)
            except Exception as e:
                st.warning(f"文件 {file.name} 处理失败: {e}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))

        # 5. 展示与导出结果
        if all_results:
            df = pd.DataFrame(all_results)
            
            # 调整列展示顺序：编号、姓名、总分在前面，源文件放最后
            base_cols = ["简历编号", "姓名", "总分"]
            end_cols = ["简评", "原文件名"]
            middle_cols = [c for c in df.columns if c not in base_cols + end_cols]
            
            final_cols = [c for c in base_cols if c in df.columns] + middle_cols + [c for c in end_cols if c in df.columns]
            df = df[final_cols]
            
            # 按总分由高到低排名
            df = df.sort_values(by="总分", ascending=False)
            
            st.success("✅ 筛选完成！")
            st.subheader(f"📊 筛选结果排名 - 选中岗位")
            st.dataframe(df, use_container_width=True)
            
            # 导出Excel按钮
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("下载表格 (CSV/Excel可直接打开)", data=csv, file_name="简历招聘排名结果.csv", mime="text/csv")