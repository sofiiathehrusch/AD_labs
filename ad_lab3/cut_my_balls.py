import streamlit as st
import pandas as pd

if 'filters' not in st.session_state:
    st.session_state.filters = {
        'selected_series': 'VCI',
        'selected_province_name': 'Kyiv',
        'selected_week_range': (1, 52),
        'selected_year_range': (1981, 2024)
    }

def load_data(province_id=None, selected_series=None):

    df = pd.read_csv("processed_data.csv", usecols=['Province_ID', 'Year', 'Week', 'VCI', 'TCI', 'VHI'])

    if province_id is not None:
        df = df[df['Province_ID'] == province_id]

    if selected_series is not None:
        df = df[['Year', selected_series, 'Province_ID', 'Week']]

    return df

controls_column, output_column = st.columns([3, 7])

with controls_column:
    time_series_options = ['VCI', 'TCI', 'VHI']
    selected_series = st.selectbox(
        "Select Time Series",
        time_series_options,
        index=time_series_options.index(st.session_state.filters['selected_series']),
    )
    st.session_state.filters['selected_series'] = selected_series

    province_mapping = {
        22: "Cherkasy", 24: "Chernihiv", 23: "Chernivtsi", 25: "Crimea", 3: "Dnipropetrovsk",
        4: "Donetsk", 8: "Ivano-Frankivsk", 19: "Kharkiv", 20: "Kherson", 21: "Khmelnytskyi",
        10: "Kirovohrad", 9: "Kyiv", 11: "Luhansk", 12: "Lviv", 13: "Mykolaiv", 14: "Odessa",
        15: "Poltava", 16: "Rivne", 17: "Sumy", 18: "Ternopil", 1: "Vinnytsia", 2: "Volyn",
        6: "Zakarpattia", 7: "Zaporizhzhia", 5: "Zhytomyr"
    }
    selected_province_name = st.selectbox(
        "Select Province",
        list(province_mapping.values()),
        index=list(province_mapping.values()).index(st.session_state.filters['selected_province_name']),
    )
    st.session_state.filters['selected_province_name'] = selected_province_name
    selected_province_id = [k for k, v in province_mapping.items() if v == st.session_state.filters['selected_province_name']][0]

    min_week, max_week = 1, 52
    selected_week_range = st.slider(
        "Select Week Interval",
        min_value=min_week,
        max_value=max_week,
        value=st.session_state.filters['selected_week_range'],
    )
    st.session_state.filters['selected_week_range'] = selected_week_range

    min_year, max_year = 1981, 2024
    selected_year_range = st.slider(
        "Select Year Interval",
        min_value=min_year,
        max_value=max_year,
        value=st.session_state.filters['selected_year_range'],
    )
    st.session_state.filters['selected_year_range'] = selected_year_range

    sort_ascending = st.checkbox("Sort Ascending")
    sort_descending = st.checkbox("Sort Descending")

    if st.button("Reset Filters"):
        st.session_state.filters = {
            'selected_series': 'VCI',
            'selected_province_name': 'Kyiv',
            'selected_week_range': (1, 52),
            'selected_year_range': (1981, 2024)
        }
        st.rerun()

with output_column:
    filtered_df = load_data(province_id=selected_province_id, selected_series=st.session_state.filters['selected_series'])
    filtered_df = filtered_df[
        (filtered_df['Week'] >= st.session_state.filters['selected_week_range'][0]) &
        (filtered_df['Week'] <= st.session_state.filters['selected_week_range'][1]) &
        (filtered_df['Year'] >= st.session_state.filters['selected_year_range'][0]) &
        (filtered_df['Year'] <= st.session_state.filters['selected_year_range'][1])
    ]

    if sort_ascending and sort_descending:
        st.warning("Both sorting options are selected. Defaulting to ascending order.")
        filtered_df = filtered_df.sort_values(by=st.session_state.filters['selected_series'], ascending=True)
    elif sort_ascending:
        filtered_df = filtered_df.sort_values(by=st.session_state.filters['selected_series'], ascending=True)
    elif sort_descending:
        filtered_df = filtered_df.sort_values(by=st.session_state.filters['selected_series'], ascending=False)

    st.write("Selected Week Range:", st.session_state.filters['selected_week_range'])
    st.write("Selected Year Range:", st.session_state.filters['selected_year_range'])
    st.write("Filtered DataFrame Shape:", filtered_df.shape)

    tab1, tab2, tab3 = st.tabs(["Filtered Data", "Filtered Data Chart", "Comparison Across Provinces"])

    with tab1:
        st.write(f"Filtered Data for {st.session_state.filters['selected_series']} in {st.session_state.filters['selected_province_name']}:")
        st.dataframe(filtered_df)

    with tab2:
        st.subheader(f"{st.session_state.filters['selected_series']} Trends Over Time")
        st.line_chart(filtered_df.set_index('Year')[st.session_state.filters['selected_series']])

    with tab3:
        st.subheader("Comparison Across Provinces")
        full_data = load_data(selected_series=st.session_state.filters['selected_series'])
        full_data = full_data[
            (full_data['Week'] >= st.session_state.filters['selected_week_range'][0]) &
            (full_data['Week'] <= st.session_state.filters['selected_week_range'][1]) &
            (full_data['Year'] >= st.session_state.filters['selected_year_range'][0]) &
            (full_data['Year'] <= st.session_state.filters['selected_year_range'][1])
        ]
        comparison_data = full_data.groupby('Province_ID')[st.session_state.filters['selected_series']].mean().reset_index()
        comparison_data['Province_Name'] = comparison_data['Province_ID'].map({k: v for k, v in province_mapping.items()})
        st.bar_chart(comparison_data.set_index('Province_Name')[st.session_state.filters['selected_series']])

