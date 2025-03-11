import os
import streamlit as st
import pandas as pd
import plotly.express as px
import json
from openai import OpenAI
#from variables import categorize_columns
import numpy as np
import anthropic
import re
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = anthropic.Anthropic(api_key=api_key)





def get_top_contributors(df, categories, segment_col):
    top_contrib_dict = {}
    for cat in categories:
        if cat in df.columns and cat != segment_col:
            # Group by category and count unique Customer IDs for each value
            top_contrib_series = df.groupby(cat)[segment_col].nunique()
            
            # Find the category value with the highest unique Customer ID count
            overall_top_contrib = top_contrib_series.idxmax()  # The most frequent category value
            top_contrib_count = top_contrib_series.max()  # Unique Customer ID count for that category value
            
            top_contrib_dict[cat] = (overall_top_contrib, top_contrib_count)
    
    return top_contrib_dict


def get_top_metric_contributors(df, categories, segment_col,metric_col,start_date,end_date,date_keys):
    df[date_keys] = pd.to_datetime(df[date_keys])
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Define six-month range
    if start_date.year == end_date.year:
        df_six_months_ago = df[df[date_keys].dt.month <= 6]
        df_current_month = df[df[date_keys].dt.month > 6]
    else:
        df_six_months_ago = df[df[date_keys].dt.year == start_date.year]
        df_current_month = df[df[date_keys].dt.year == end_date.year]

    # Initialize status dictionary and Metric Contribution DataFrame
    status_dict = {}
    Metric_Contribution = pd.DataFrame(columns=['Category', 'Contributor', 'Count', 'Type'])

    # Ensure Action_Name contains valid column names
    for met in categories:  # Assuming 'Action_Name' lists metric names
        if met not in df.columns:
            continue  # Skip invalid metric names

        # Aggregate previous and current period metrics
        six_months_summary = df_six_months_ago.groupby(segment_col)[met].sum().reset_index()
        current_month_summary = df_current_month.groupby(segment_col)[met].sum().reset_index()

        # Rename columns for clarity
        six_months_summary.rename(columns={met: f'prev_{met}'}, inplace=True)
        current_month_summary.rename(columns={met: f'curr_{met}'}, inplace=True)

        # Merge for comparison
        if met == categories[0]:  # First metric, create df_comparison
            df_comparison = pd.merge(six_months_summary, current_month_summary, on=segment_col, how='inner')
        else:
            df_comparison = pd.merge(df_comparison, six_months_summary, on=segment_col, how='inner')
            df_comparison = pd.merge(df_comparison, current_month_summary, on=segment_col, how='inner')

        # Calculate percentage change
        df_comparison[f'{met}_Change_%'] = (
            (df_comparison[f'curr_{met}'] - df_comparison[f'prev_{met}']) / df_comparison[f'prev_{met}']
        ) * 100

        # Handle NaN and infinite values
        df_comparison.replace([float('inf'), float('-inf')], 0, inplace=True)
        df_comparison[f'{met}_Change_%'].fillna(0, inplace=True)

        # Determine trend status
        df_comparison[f'{met}_label'] = df_comparison[f'{met}_Change_%'].apply(
            lambda x: 'Decreasing' if x < 0 else ('Increasing' if x > 0 else 'No Change')
        )
            # Count the number of contributors for each trend
    for met in categories:
    # Only process if `met` is not equal to `metric_keys`
        if met != metric_col:
            metric_label = metric_col + '_label'  # Single metric key assumed
            print(f"Comparing '{met}' with '{metric_col}'")
            # Compute contributor counts
            pos_inc = df_comparison[(df_comparison[metric_label] == 'Increasing') & (df_comparison[met+'_label'] == 'Increasing')].shape[0]
            neg_inc = df_comparison[(df_comparison[metric_label] == 'Decreasing') & (df_comparison[met+'_label'] == 'Increasing')].shape[0]
            pos_dec = df_comparison[(df_comparison[metric_label] == 'Increasing') & (df_comparison[met+'_label'] == 'Decreasing')].shape[0]
            neg_dec = df_comparison[(df_comparison[metric_label] == 'Decreasing') & (df_comparison[met+'_label'] == 'Decreasing')].shape[0]

            # Append results to Metric_Contribution
            Metric_Contribution = pd.concat([
                Metric_Contribution,
                pd.DataFrame([
                    {'Category': met, 'Contributor': 'Increasing', 'Count': pos_inc, 'Type': f'{metric_col} Increasing'},
                    {'Category': met, 'Contributor': 'Increasing', 'Count': neg_inc, 'Type': f'{metric_col} Decreasing'},
                    {'Category': met, 'Contributor': 'Decreasing', 'Count': pos_dec, 'Type': f'{metric_col} Increasing'},
                    {'Category': met, 'Contributor': 'Decreasing', 'Count': neg_dec, 'Type': f'{metric_col} Decreasing'}
                ])
            ], ignore_index=True)
    Metric_Contribution["Category_value"] = Metric_Contribution["Category"] + " (" + Metric_Contribution["Contributor"] + ")"

    # Group by Category_value and Type, summing Count
    grouped_df = Metric_Contribution.groupby(["Category_value", "Type"])["Count"].sum().reset_index()

    # Calculate percentage change
    grouped_df["Change %"] = grouped_df.groupby("Category_value")["Count"].pct_change().abs() * 100

    # Find the highest percentage change for each Category
    category_changes = []
    for category in Metric_Contribution["Category"].unique():
        category_df = grouped_df[grouped_df["Category_value"].str.startswith(category)]
        
        if not category_df["Change %"].isna().all():
            max_change_row = category_df.loc[category_df["Change %"].idxmax()]
            category_changes.append(max_change_row["Category_value"])

    # Filter records from the original DataFrame
    filtered_df = Metric_Contribution[Metric_Contribution["Category_value"].isin(category_changes)].copy()   
    return {"Top Metric": filtered_df.copy(), "Metric_Change_%": df_comparison.copy()}



def pos_neg_list(df, neg_sponsor, pos_sponsor, Seg_Col,Metric_Col,Date_Col,start_date,end_date):
    df = df.copy()
    #st.dataframe(df)
    
    # Ensure 'Start Date' is in datetime format
    df[Date_Col] = pd.to_datetime(df[Date_Col])
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)



    #st.write(df.shape)
    # Filter data for the past and current year
    
    if start_date.year==end_date.year:
        df_six_months_ago = df[(df[Date_Col].dt.month <=6)]# & ( df[Date_Col].dt.day <=14)  ]
        df_current_month = df[(df[Date_Col].dt.month >6) ]#& ( df[Date_Col].dt.day >14)  ]
    else:
        df_six_months_ago = df[df[Date_Col].dt.year == start_date.year]
        df_current_month = df[df[Date_Col].dt.year == end_date.year]

    # st.write("Data for year"+str(start_date.year)+str(df_six_months_ago.shape))
    # st.write("Data for year"+str(end_date.year)+str(df_current_month.shape))

    # Aggregate enrollment sum by sponsor
    
    six_months_summary = df_six_months_ago.groupby(Seg_Col)[Metric_Col].sum().reset_index(name='prev_Metric')
    current_month_summary = df_current_month.groupby(Seg_Col)[Metric_Col].sum().reset_index(name='curr_Metric')

    # Merge summaries
    df_comparison = pd.merge(six_months_summary, current_month_summary, on=Seg_Col, how='inner')

    # Calculate percentage change
    df_comparison['Metric_Change_%'] = (
        (df_comparison['curr_Metric'] - df_comparison['prev_Metric']) / df_comparison['prev_Metric']
    ) * 100

    # Replace infinite values with 0
    df_comparison.replace([float('inf'), float('-inf')], 0, inplace=True)
    df_comparison['Metric_Change_%'].fillna(0,inplace=True)

    #st.dataframe(df_comparison)

    # Update sponsor lists
    neg_sponsor.extend(df_comparison.loc[df_comparison['Metric_Change_%'] < 0, Seg_Col].tolist())
    pos_sponsor.extend(df_comparison.loc[df_comparison['Metric_Change_%'] > 0, Seg_Col].tolist())


# Load Data
data = pd.read_csv(r"UniqueHCPDigitalData.csv")

date_columns = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]

#st.write(date_columns)
#date_col=date_columns[0]

data['Action_Date'] = pd.to_datetime(data['Action_Date'])  # Ensure datetime format

neg_sponsor = []
pos_sponsor = []
df_neg=pd.DataFrame()
df_pos=pd.DataFrame()


# # Display data preview
# with st.expander("Data Preview", expanded=False):
#     st.dataframe(data)

# Create columns for dropdowns
col1, col2, col3, col4 = st.columns([1, 1,1,1])  # Two columns for Segment & Metric
#col3, col4 = st.columns([1, 1])  # Two columns for Start & End Date

with col1:
    st.markdown("**Segment**")  
    segment_col = st.selectbox("", options=data.select_dtypes(include=["object"]).columns, key="segment")

with col2:
    st.markdown("**Metric**")
    metric_col = st.selectbox("", options=data['Action_Name'].unique(), key="metric")

# Date Selection  
with col3:
    st.markdown("**Start Date**")
    start_date = st.date_input("", min(data["Action_Date"]), key="start_date")

with col4:
    st.markdown("**End Date**")
    end_date = st.date_input("", max(data["Action_Date"]), key="end_date")

# Validate Date Selection
if start_date > end_date:
    st.error("⚠️ End Date must be greater than or equal to Start Date.")

# Display selected filters
#st.write(f"**Selected Filters:** Segment - {segment_col}, Metric - {metric_col}, Date Range: {start_date} to {end_date}")
if segment_col and metric_col and start_date and end_date:
    start_date=pd.to_datetime(start_date)
    end_date=pd.to_datetime(end_date)
    
    All_group_df = None

    for met in data['Action_Name'].unique():
        filtered_df = data[data['Action_Name'] == met].copy()

        group_df = filtered_df.groupby([segment_col, 'Action_Date']).size().reset_index(name=str(met))

        if All_group_df is None:
            All_group_df = group_df  # Initialize with the first iteration
        else:
            All_group_df = pd.merge(All_group_df, group_df, how='outer', on=[segment_col, 'Action_Date'])

    All_group_df.fillna(0, inplace=True)
    #st.dataframe(All_group_df)
    if start_date.year==end_date.year:
            pos_neg_list(All_group_df, neg_sponsor, pos_sponsor, segment_col,metric_col,'Action_Date',start_date,end_date)
    result=get_top_metric_contributors(All_group_df, data['Action_Name'].unique(), segment_col,metric_col,start_date,end_date,'Action_Date')
    Metric_Contribution= result["Top Metric"]
    metric_change_df = result["Metric_Change_%"]
    # st.dataframe(Metric_Contribution)
    
    
    # st.subheader("Positive "+segment_col)
    # st.dataframe(pd.DataFrame({"Positive "+segment_col: pos_sponsor}))

    # st.subheader("Negative "+segment_col)
    # st.dataframe(pd.DataFrame({"Negative "+segment_col: neg_sponsor}))
    
    
    categories = ['HCP_Specialities','Brands','Channels','Campaign Type','Publishers','Device','Adopters','Campaigning Name']
    df_pos = data[
        (data[segment_col].isin(pos_sponsor)) &
        (data['Action_Date'] >= start_date) &
        (data['Action_Date'] <= end_date)
    ]

    df_neg = data[
        (data[segment_col].isin(neg_sponsor)) &
        (data['Action_Date'] >= start_date) &
        (data['Action_Date'] <= end_date)
    ]
    
    # st.subheader("Positive Segment Data")
    # st.dataframe(df_pos)

    # st.subheader("Negative Segment Data")
    # st.dataframe(df_neg)
    
    top_contributors_pos={}
    top_contributors_neg={}
    if not df_neg.empty:
    #categories=categorical_cols
        if df_pos.shape[0]==0 and df_neg.shape[0]==0:
            st.warning("No data is present in that timeframe")
        else:
            # Get the top contributor for each category for both positive and negative data
            top_contributors_pos = get_top_contributors(df_pos, categories,segment_col)
            top_contributors_neg = get_top_contributors(df_neg, categories,segment_col)

            # st.write(top_contributors_pos)
            # st.write(top_contributors_neg)

        ### Only selecting Common Columns
            common_keys = {
                key for key in top_contributors_pos
                if key in top_contributors_neg and top_contributors_pos[key][0] == top_contributors_neg[key][0]
            }
                    
        ###Removing ID and single value Columns
            for col in data.columns:
                if col in common_keys:
                    unique_values = data[col].nunique()
                    if unique_values == len(data[col]) or unique_values == 1:
                        common_keys.remove(col)      
                
        top_contributors_pos = {key: top_contributors_pos[key] for key in common_keys}
        top_contributors_neg = {key: top_contributors_neg[key] for key in common_keys}
        
    
    

# === Visualization ===
if top_contributors_pos and top_contributors_neg:
    # Create a DataFrame for plotting from the contributor dictionaries
    contrib_df = pd.DataFrame({
        'Category': list(top_contributors_pos.keys()),
        'Positive Contributor': [value[0] for value in top_contributors_pos.values()],
        'Positive Count': [value[1] for value in top_contributors_pos.values()],
        'Negative Contributor': [top_contributors_neg.get(cat, ('No Data', 0))[0] for cat in top_contributors_pos.keys()],
        'Negative Count': [top_contributors_neg.get(cat, ('No Data', 0))[1] for cat in top_contributors_pos.keys()],
    })
    
    # Calculate the absolute difference between positive and negative counts
    contrib_df['Diff'] = np.abs(contrib_df['Positive Count'] - contrib_df['Negative Count'])

    # Order the categories based on difference (descending order)
    contrib_df = contrib_df.sort_values('Diff', ascending=False)
    ordered_categories = contrib_df['Category'].tolist()

    # Create a DataFrame for positive contributions
    df_positive = contrib_df[['Category', 'Positive Contributor', 'Positive Count']].copy()
    df_positive = df_positive.rename(columns={'Positive Contributor': 'Contributor', 'Positive Count': 'Count'})
    df_positive['Type'] = metric_col+' Increasing'

    # Create a DataFrame for negative contributions
    df_negative = contrib_df[['Category', 'Negative Contributor', 'Negative Count']].copy()
    df_negative = df_negative.rename(columns={'Negative Contributor': 'Contributor', 'Negative Count': 'Count'})
    df_negative['Type'] = metric_col+' Decreasing'

    # Combine the two DataFrames
    contrib_df_melted = pd.concat([df_positive, df_negative], ignore_index=True)
    

    # Create a new ordered Category_value column
    contrib_df_melted['Category_value'] = contrib_df_melted['Category'] + " (" + contrib_df_melted['Contributor'] + ")"
    contrib_df_melted=pd.concat([contrib_df_melted,Metric_Contribution],ignore_index=True)
    #st.dataframe(contrib_df_melted)
 

    df_grouped = contrib_df_melted.groupby(['Category_value', 'Type'])['Count'].sum().unstack(fill_value=0)
    df_grouped['Change'] = df_grouped.get(metric_col+' Increasing', 0) - df_grouped.get(metric_col+' Decreasing', 0)

    # Filter rows where absolute difference is > 5
    df_filtered = df_grouped[abs(df_grouped['Change']) > 1]

    # Sort by absolute difference (descending)
    df_filtered = df_filtered.sort_values(by='Change', ascending=True)

    # Extract ordered category list
    sorted_categories = df_filtered.index.tolist()

    # Filter original DataFrame based on sorted categories
    filtered_df = contrib_df_melted[contrib_df_melted['Category_value'].isin(sorted_categories)].copy()

    # Set categorical order for sorting
    filtered_df['Category_value'] = pd.Categorical(filtered_df['Category_value'], categories=sorted_categories, ordered=True)

    # Ensure correct sorting in filtered_df
    filtered_df = filtered_df.sort_values(by='Category_value', key=lambda x: x.cat.codes)

    # Display DataFrame in Streamlit
    #st.dataframe(filtered_df)
    ## Plot if data exists
    
    if not metric_change_df.empty:
        fig = px.scatter(
            metric_change_df, 
            x=metric_col+"_Change_%", 
            y="curr_"+metric_col, 
            title=segment_col+" Plot",
            labels={metric_col+"_Change_%": "% "+metric_col+" Change", "curr_"+metric_col: "Total "+metric_col},
            size_max=10
        )
        
        # Set dot color to light blue
        fig.update_traces(marker=dict(color='skyblue'))
        
        # Ensure x-axis starts at 0 but extends left if negative values exist
        min_x = metric_change_df[metric_col+"_Change_%"].min()
        max_x = metric_change_df[metric_col+"_Change_%"].max()
        fig.update_layout(
            title_x=0.4,
            xaxis=dict(
                zeroline=True, 
                zerolinecolor='black',
                range=[min(0, min_x), max_x]  # Ensures 0 is included unless all values are positive
            )
        )
        
        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Scatter chart did not plot")  


    if not filtered_df.empty:
        fig = px.bar(
            filtered_df,
            x='Category_value',  
            y='Count',
            color='Type',
            text='Contributor',
            title="Top Contributors by Category",
            labels={'Category_value': 'Category', 'Count': segment_col+' Count', 'Type': 'Change Type'},
            barmode='group',
            text_auto=True,
            category_orders={"Type": [metric_col+" Increasing", metric_col+" Decreasing"]},  # Forces Decreasing First
            color_discrete_map={metric_col+" Decreasing": "lightcoral", metric_col+" Increasing": "skyblue"}  # 🔹 Color Mapping
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No categories meet the >10% change threshold.")

    
    #with col2:
        #st.dataframe(metric_change_df)
    

        
if top_contributors_pos and top_contributors_neg:
    prompt = f"""

    You are an expert data analyst.

    Below is a DataFrame:

    ### DataFrame:
    {contrib_df_melted}

    Summarize each category with bullet points, ensuring each summary is exactly one line. 
    Only include categories where the percentage change is greater than 30%. 
    
    The summary should include:
    - The category name.
    - The contributor's value described in words.
    - The percentage change shoud be written in number with bold letters.
    - The impact on the total count, stating whether it is increasing or decreasing.

    Return only the summary in bullets points.
    """
    response = client.chat.completions.create(
       model="gpt-4-turbo",
       messages=[{"role": "system", "content": "You are a data analysis assistant."},
                   {"role": "user", "content": prompt}]
    )
    
    conversation = [{"role": "user", "content": prompt}]
    response = client.messages.create(
                    model='claude-3-5-sonnet-20241022',
                    messages=conversation,
                    max_tokens=5000
                )

    lines = response.content[0].text.split('\n')
    for line in lines:
        if line != "":
            st.write(line.strip())
