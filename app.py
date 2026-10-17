import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

st.set_page_config(
    page_title="AI Data Analyst Agent", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-card {
        background: rgba(26, 29, 46, 1);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .cyan-glow { border-top: 4px solid #00F2FE; }
    .purple-glow { border-top: 4px solid #BF5AF2; }
    .pink-glow { border-top: 4px solid #FF375F; }
    .gold-glow { border-top: 4px solid #FFD60A; }

    .metric-title {
        color: #8A8D9F;
        font-size: 13px;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 30px;
        font-weight: 700;
    }
    .section-header {
        font-size: 20px;
        font-weight: 650;
        color: #FFFFFF;
        margin-bottom: 15px;
        margin-top: 20px;
    }
    .wizard-box {
        background: rgba(16, 18, 30, 0.8);
        border: 1px solid rgba(0, 242, 254, 0.3);
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 15px;">
        <div>
            <h1 style="margin:0; font-size: 2.3rem; font-weight: 800; background: linear-gradient(to right, #00F2FE, #4FACFE); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🤖 AI Data Analyst Agent</h1>
            <p style="margin: 3px 0 0 0; color: #8A8D9F; font-size: 15px;">Transforming RAW data into automated predictive insights & visualizations.</p>
        </div>
        <div style="background: rgba(0,242,254,0.1); border: 1px solid rgba(0,242,254,0.2); border-radius: 30px; padding: 6px 16px; font-weight:600; color:#00F2FE; font-size:13px;">
            💎 PRO VERSION
        </div>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Agent Configuration")
    api_key = st.text_input("Enter OpenAI API Key", type="password", help="Required for Autonomous Reporting & Custom Commands")
    
    st.markdown("---")
    st.markdown("### 🛠️ Active Tech Stack")
    st.markdown("- 🐍 **Python** (Backbone)\n- 🐼 **Pandas** (ETL Pipeline)\n- 📊 **Plotly** (Dynamic Render)\n- 🌐 **Streamlit** (Cyber Front)\n- 🤖 **OpenAI API** (LLM Cognitive)")

client = OpenAI(api_key=api_key) if api_key else None


def thorough_auto_clean(df):
    """Deep automated cleaning engine."""
    cleaned_df = df.copy()
    
    # 1. Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # 2. Identify & Drop useless ID/Sequential columns (e.g., PassengerId, Unnamed: 0)
    cols_to_drop = []
    for col in cleaned_df.columns:
        col_lower = col.lower()
        if "id" in col_lower or "unnamed" in col_lower or "index" in col_lower:
            if cleaned_df[col].nunique() == len(cleaned_df):
                cols_to_drop.append(col)
    
    if cols_to_drop:
        cleaned_df = cleaned_df.drop(columns=cols_to_drop)
        
    # 3. Clean strings & coerce text numbers / currency
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == 'object':
            # Strip whitespace
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
            
            # Currency / Numeric parsing attempt
            temp_cleaned = cleaned_df[col].str.replace(r'[$,£€,]', '', regex=True)
            temp_numeric = pd.to_numeric(temp_cleaned, errors='coerce')
            
            valid_orig = cleaned_df[col].notna().sum()
            valid_num = temp_numeric.notna().sum()
            
            if valid_orig > 0 and (valid_num / valid_orig) > 0.5:
                cleaned_df[col] = temp_numeric

    # 4. Handle remaining missing values intelligently
    for col in cleaned_df.columns:
        if cleaned_df[col].isna().sum() > 0:
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            else:
                mode_val = cleaned_df[col].mode()
                cleaned_df[col] = cleaned_df[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")
                
    return cleaned_df


def apply_ai_custom_instruction(df, user_instruction, client):
    prompt = f"""
    You are an expert Python Data Analyst.
    Given a pandas DataFrame named `df` with columns: {list(df.columns)}
    
    Write executable Python pandas code to satisfy this user requirement: "{user_instruction}"
    
    Requirements:
    - Modify or assign to `df`.
    - Do NOT wrap in markdown backticks or python code blocks. Output ONLY raw executable Python code.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    code = response.choices[0].message.content.strip()
    
    local_vars = {"df": df.copy(), "pd": pd}
    exec(code, {}, local_vars)
    return local_vars["df"]


if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
if "selected_graph_type" not in st.session_state:
    st.session_state.selected_graph_type = "Bar Graph"

st.markdown('<div class="section-header">📁 Dataset Upload</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")

if uploaded_file is not None:
    if st.session_state.uploaded_filename != uploaded_file.name:
        st.session_state.raw_df = pd.read_csv(uploaded_file).dropna(axis=1, how='all')
        st.session_state.cleaned_df = None
        st.session_state.uploaded_filename = uploaded_file.name

    raw_df = st.session_state.raw_df

    if st.session_state.cleaned_df is None:
        st.markdown('<div class="wizard-box">', unsafe_allow_html=True)
        st.markdown('<h3 style="color:#00F2FE; margin-top:0;">🪄 Interactive Data Cleaning Wizard</h3>', unsafe_allow_html=True)
        st.write("Configure your cleaning strategy below:")
        
        missing_cols = raw_df.columns[raw_df.isna().any()].tolist()
        
        with st.form("cleaning_form"):
            cleaning_prefs = {}
            
            if missing_cols:
                st.markdown("#### 🚨 Missing Values Detected")
                for col in missing_cols:
                    null_count = raw_df[col].isna().sum()
                    
                    options = ["Let AI Decide (Comprehensive Auto-Clean)", "Do Nothing", "Drop Rows with Missing Values"]
                    if pd.api.types.is_numeric_dtype(raw_df[col]):
                        options.extend(["Fill with Mean", "Fill with Median", "Fill with 0"])
                    else:
                        options.extend(["Fill with 'Unknown'", "Fill with Mode"])
                        
                    cleaning_prefs[col] = st.selectbox(
                        f"Column '{col}' has {null_count} missing values. Action:", 
                        options
                    )
            else:
                st.success("✅ No missing values detected in this dataset!")
            
            st.markdown("#### 🔤 Data Type & Structure Optimization")
            auto_ai_clean_all = st.checkbox("Enable Full AI Auto-Clean (Drop ID Columns, Deduplicate, Parse Currencies, Impute Nulls)", value=True)
            
            st.markdown("#### 📈 Select Default Dashboard Graph")
            graph_choice = st.selectbox(
                "Which interactive graph type should be generated initially?",
                ["Bar Graph", "Line Graph", "Pie Chart", "Histogram", "Scatter Plot", "Area Chart", "Treemap", "Funnel Chart"]
            )
            
            st.markdown("#### ✍️ Custom User Instructions")
            user_instruction = st.text_input(
                "Enter custom cleaning or transformation requests (e.g., 'Filter rows where Age > 20'):", 
                placeholder="Type your request here..."
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit_cleaning = st.form_submit_button("🧹 Clean Data & Generate Dashboard")
            
            if submit_cleaning:
                temp_df = raw_df.copy()
                st.session_state.selected_graph_type = graph_choice
                
                # Apply column-by-column user preferences
                for col, action in cleaning_prefs.items():
                    if action == "Drop Rows with Missing Values":
                        temp_df = temp_df.dropna(subset=[col])
                    elif action == "Fill with Mean":
                        temp_df[col] = temp_df[col].fillna(temp_df[col].mean())
                    elif action == "Fill with Median":
                        temp_df[col] = temp_df[col].fillna(temp_df[col].median())
                    elif action == "Fill with 0":
                        temp_df[col] = temp_df[col].fillna(0)
                    elif action == "Fill with 'Unknown'":
                        temp_df[col] = temp_df[col].fillna('Unknown')
                    elif action == "Fill with Mode":
                        mode_val = temp_df[col].mode()
                        temp_df[col] = temp_df[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")
                
                # Apply Full AI Auto-Clean if toggled or selected
                if auto_ai_clean_all:
                    temp_df = thorough_auto_clean(temp_df)
                
                # Custom AI prompt execution
                if user_instruction.strip():
                    if client:
                        try:
                            temp_df = apply_ai_custom_instruction(temp_df, user_instruction, client)
                            st.toast("Custom AI Transformation Applied Successfully!", icon="✨")
                        except Exception as e:
                            st.error(f"Failed to apply custom instruction: {e}")
                    else:
                        st.warning("⚠️ Enter your OpenAI API Key in the sidebar to execute custom text instructions.")

                st.session_state.cleaned_df = temp_df
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        df = st.session_state.cleaned_df
        neon_colors = ["#00F2FE", "#BF5AF2", "#FFD60A", "#FF375F", "#00FF66", "#FF9F1C", "#38B6FF", "#FF5722"]
        
        c1, c2 = st.columns([8, 2])
        with c1:
            st.success("✨ Dataset cleaned successfully according to your parameters!")
        with c2:
            if st.button("🔄 Reset & Re-Clean"):
                st.session_state.cleaned_df = None
                st.rerun()

        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Cleaned Dataset (CSV)",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
        )
        
        st.markdown('<div class="section-header">📊 Automatic Data Profiling</div>', unsafe_allow_html=True)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.markdown(f'<div class="metric-card cyan-glow"><div class="metric-title">📂 Total Rows</div><div class="metric-value">{df.shape[0]:,}</div></div>', unsafe_allow_html=True)
        kpi2.markdown(f'<div class="metric-card purple-glow"><div class="metric-title">📐 Total Columns</div><div class="metric-value">{df.shape[1]}</div></div>', unsafe_allow_html=True)
        kpi3.markdown(f'<div class="metric-card pink-glow"><div class="metric-title">🔍 Remaining Missing</div><div class="metric-value">{df.isna().sum().sum():,}</div></div>', unsafe_allow_html=True)
        kpi4.markdown(f'<div class="metric-card gold-glow"><div class="metric-title">📋 Duplicate Entries</div><div class="metric-value">{df.duplicated().sum()}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">🔍 Preview Cleaned Data</div>', unsafe_allow_html=True)
        st.dataframe(df.head(7), use_container_width=True)

        st.markdown('<div class="section-header">📈 Dynamic Multi-Color Visual Explorer</div>', unsafe_allow_html=True)
        
        all_cols = df.columns.tolist()
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        c_x, c_y, c_type, c_color = st.columns(4)
        with c_x:
            x_axis = st.selectbox("X-Axis (Main Category)", all_cols, index=0)
        with c_y:
            y_options = num_cols if num_cols else all_cols
            y_axis = st.selectbox("Y-Axis (Values)", y_options, index=0 if y_options else None)
        with c_type:
            graph_type = st.selectbox(
                "Change Graph Type",
                ["Bar Graph", "Line Graph", "Pie Chart", "Histogram", "Scatter Plot", "Area Chart", "Treemap", "Funnel Chart"],
                index=["Bar Graph", "Line Graph", "Pie Chart", "Histogram", "Scatter Plot", "Area Chart", "Treemap", "Funnel Chart"].index(st.session_state.selected_graph_type)
            )
        with c_color:
            color_opt = ["Auto Multi-Color", "Single Neon Color"] + all_cols
            color_axis = st.selectbox("Color Segment", color_opt, index=0)
            
        color_kwarg = x_axis if color_axis == "Auto Multi-Color" else (None if color_axis == "Single Neon Color" else color_axis)
        
        fig = None

        # Prepare Aggregated / Cleaned Data for High-Cardinality Pie/Bar Charts to prevent overlap
        plot_df = df.copy()
        if graph_type in ["Pie Chart", "Bar Graph", "Funnel Chart"] and plot_df[x_axis].nunique() > 10:
            top_10 = plot_df[x_axis].value_counts().nlargest(10).index
            plot_df[x_axis] = plot_df[x_axis].apply(lambda v: v if v in top_10 else "Other")

        try:
            if graph_type == "Bar Graph":
                fig = px.bar(plot_df, x=x_axis, y=y_axis, color=color_kwarg, title=f"Bar Graph: Top Categories in {x_axis}", color_discrete_sequence=neon_colors)
            elif graph_type == "Line Graph":
                fig = px.line(df, x=x_axis, y=y_axis, color=color_kwarg, title=f"Line Graph: {x_axis} vs {y_axis}", color_discrete_sequence=neon_colors)
            elif graph_type == "Pie Chart":
                fig = px.pie(plot_df, names=x_axis, values=y_axis if y_axis in num_cols else None, color=color_kwarg, title=f"Cleaned Pie Chart: {x_axis} Distribution (Top 10)", color_discrete_sequence=neon_colors)
                fig.update_traces(textposition='inside', textinfo='percent+label')
            elif graph_type == "Histogram":
                fig = px.histogram(df, x=x_axis, color=color_kwarg, title=f"Histogram: {x_axis}", marginal="box", color_discrete_sequence=neon_colors)
            elif graph_type == "Scatter Plot":
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_kwarg, title=f"Scatter Plot: {x_axis} vs {y_axis}", color_discrete_sequence=neon_colors)
            elif graph_type == "Area Chart":
                fig = px.area(df, x=x_axis, y=y_axis, color=color_kwarg, title=f"Area Chart: {x_axis} vs {y_axis}", color_discrete_sequence=neon_colors)
            elif graph_type == "Treemap":
                if y_axis in num_cols:
                    fig = px.treemap(plot_df, path=[x_axis], values=y_axis, color=color_kwarg, title=f"Treemap: {x_axis} sized by {y_axis}", color_discrete_sequence=neon_colors)
                else:
                    st.warning("Treemap requires a numerical Y-axis for sizing.")
            elif graph_type == "Funnel Chart":
                fig = px.funnel(plot_df, x=y_axis, y=x_axis, color=color_kwarg, title=f"Funnel Chart: {x_axis}", color_discrete_sequence=neon_colors)
            
            if fig:
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    font_color="#8A8D9F",
                    title_font=dict(size=20, color="#FFFFFF"),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                if color_axis == "Single Neon Color" and graph_type not in ["Pie Chart", "Treemap"]:
                    fig.update_traces(marker_color="#00F2FE", line_color="#00F2FE")
                    
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Cannot render {graph_type} with the selected axis combination. Try selecting numerical values for the Y-axis. Error: {e}")

        st.markdown('<div class="section-header">🧠 Autonomous AI Insight Agent</div>', unsafe_allow_html=True)
        if not api_key:
            st.info("🔑 Feed an OpenAI authentication key in the sidebar panel to unlock automated LLM Audit generation.")
        else:
            if st.button("🚀 Invoke AI Audit Agent"):
                with st.spinner("🤖 Agent pipeline is analyzing statistics & mapping anomalies..."):
                    try:
                        prompt = f"""
                        Analyze this dataset. Provide a highly structured response with:
                        1. Executive Insight
                        2. Data Quality Audit
                        3. Key Statistical Trends
                        4. Actionable Recommendations
                        
                        Head:\n{df.head(5).to_string()}
                        Stats:\n{df.describe().to_string()}
                        Nulls:\n{df.isna().sum().to_string()}
                        """
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.4
                        )
                        report = response.choices[0].message.content
                        st.markdown("### 📄 Generation Output")
                        st.write(report)
                        st.download_button("📥 Save Report (.md)", data=report, file_name="Agent_Audit.md")
                    except Exception as e:
                        st.error(f"Execution Error: {e}")