import streamlit as st
import pandas as pd
from pdfminer.high_level import extract_text
from openai import OpenAI
import json

# 页面配置
st.set_page_config(page_title="货代HR简历筛选助手", layout="wide")
st.title("🚢 货代行业简历 AI 智能筛选系统")

# 侧边栏：配置密钥
with st.sidebar:
    st.header("系统设置")
    api_key = st.text_input("请输入 API Key (DeepSeek/OpenAI)", type="password")
    base_url = st.text_input("API 接口地址", value="https://api.deepseek.com")
    model_name = st.text_input("模型名称", value="deepseek-chat")

# 评分标准（固化在程序里，免去重复输入）
SCORING_PROMPT = """
你是一名资深货代HR。请根据以下标准打分（满分100分）：
1. 求职状态 (15分)：离职随时到岗(15) / 在职看机会(10) / 暂不急跳槽(5)。
2. 相关行业经验 (40分)：>5年同岗位(40) / >5年相关行业(35) / 3-5年同岗位(30) / 3-5年相关行业(25)。
3. 企业背景 (30分)：世界500强/知名大货代(30) / 中型规范企业(20) / 行业相关公司(15)。
4. 学历排序 (15分)：双一流/海事类名校(集美、上海海事、大连海事等)(15) / 普通本科(10) / 专科(5)。
请输出JSON格式：{"姓名": "", "总分": 0, "维度得分": {"状态":0, "经验":0, "背景":0, "学历":0}, "简评": ""}
"""

# 上传文件
uploaded_files = st.file_uploader("请上传求职者 PDF 简历（支持多选）", type="pdf", accept_multiple_files=True)

if st.button("开始 AI 筛选排名") and uploaded_files:
    if not api_key:
        st.error("请先在左侧输入 API Key！")
    else:
        client = OpenAI(api_key=api_key, base_url=base_url)
        all_results = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            # 1. 提取文本
            text = extract_text(file)
            
            # 2. 调用 AI
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": f"{SCORING_PROMPT}\n简历内容：\n{text[:3000]}"}],
                    response_format={'type': 'json_object'}
                )
                res_json = json.loads(response.choices[0].message.content)
                all_results.append(res_json)
            except Exception as e:
                st.warning(f"文件 {file.name} 处理失败: {e}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))

        # 3. 展示结果
        df = pd.DataFrame(all_results)
        df = df.sort_values(by="总分", ascending=False)
        
        st.subheader("📊 筛选结果排名")
        st.dataframe(df, use_container_width=True)
        
        # 4. 下载按钮
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("导出 Excel 表格", data=csv, file_name="简历评分结果.csv", mime="text/csv")