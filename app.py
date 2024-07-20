import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(
    page_title="Qimmatbaho olmoslar tahlili",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data():
    df = pd.read_csv('16_1.csv')
    df = df.drop_duplicates()
    return df

df = load_data()

st.title("💎 Qimmatbaho olmoslar tahlili")
st.markdown("---")

st.sidebar.markdown("## 🔍 Bo'limlar")
st.sidebar.markdown("---")



category = st.sidebar.selectbox(
    "Kategoriya tanlang:",
    ["Data set haqida umumiy ma'lumot", "Statistika", "Grafiklar", "Qisqa Tahlillar", "Xulosa"]
)

if category == "Data set haqida umumiy ma'lumot":
    
    st.header("Qimmmatbaho olmoslar")
    st.write("Bu qimmatbaho toshlar ma'lumotlar to'plami. Ma'lumotlar quyidagi ustunlarni o'z ichiga oladi: `carat`, `cut`, `color`, `clarity`, `price`, `x`, `y`, `z`, `country`, `years`, `sold`.")
    st.write(df)
    st.image("https://i.pinimg.com/originals/0b/46/97/0b4697db0d8b95ebfff5c5f4ff8ee217.jpg")
elif category == "Statistika":
    st.write('## Descriptive Statistics')
    st.write(df.describe())
    st.header("Raqamli ma'lumotlar korralatsiyasi: ")
    options = st.multiselect('Select multiple options to display:',df.select_dtypes("number").columns,default="x")
    correlation_matrix = df[options].corr()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    st.pyplot(plt)


elif category == "Grafiklar":
    
    st.header("Filtrlar")
# Filterlar
    if 'carat' in df.columns:
        carat_min, carat_max = float(df['carat'].min()), float(df['carat'].max())
        selected_carat = st.sidebar.slider('Carat', carat_min, carat_max, (carat_min, carat_max))

    if 'cut' in df.columns:
        cuts = df['cut'].unique().tolist()
        selected_cuts = st.sidebar.multiselect('Cut', cuts, cuts)

    if 'color' in df.columns:
        colors = df['color'].unique().tolist()
        selected_colors = st.sidebar.multiselect('Color', colors, colors)

    if 'clarity' in df.columns:
        clarities = df['clarity'].unique().tolist()
        selected_clarities = st.sidebar.multiselect('Clarity', clarities, clarities)

    if 'country' in df.columns:
        countries = df['country'].unique().tolist()
        selected_countries = st.sidebar.multiselect('Country', countries, countries)

    if 'years' in df.columns:
        years = df['years'].unique().tolist()
        selected_years = st.sidebar.multiselect('Years', years, years)

    # Filtrlangan ma'lumotlar
    filtered_data = df[
        (df['carat'] >= selected_carat[0]) & (df['carat'] <= selected_carat[1]) &
        (df['cut'].isin(selected_cuts)) &
        (df['color'].isin(selected_colors)) &
        (df['clarity'].isin(selected_clarities)) &
        (df['country'].isin(selected_countries)) &
        (df['years'].isin(selected_years))
    ]

    st.write(filtered_data)

    # Ma'lumotlarni ko'rish yoki statistik tahlilni tanlash
    option = st.radio('Choose a view:', ['Data Preview', 'Descriptive Statistics'])

    if option == 'Data Preview':
        st.write('## Data Preview')
        st.dataframe(filtered_data.head())
    else:
        st.write('## Descriptive Statistics')
        st.write(filtered_data.describe())

    # Grafiklar uchun tablar
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab8= st.tabs([
            'Miqdoriy o\'zgaruvchilar uchun tavsifiy statistika', 
            'Miqdoriy o\'zgaruvchilar korrelyatsiyasi', 
            'Yo\'q ma\'lumotlar', 
            'To\'ldirishdan oldingi va keyingi taqsimotlar', 
            'Yillik sotuvlar va o\'rtacha narx dinamikasi',
            'Outlierslarni aniqlash va olib tashlash',
            'Kesim (cut) bo\'yicha narx taqsimoti'])

    with tab1:
            # Miqdoriy o'zgaruvchilarni tanlab olamiz
            numeric_columns = ['carat', 'price', 'x', 'y', 'z', 'sold']

            # Tavsifiy statistikani hisoblaymiz
            desc_stats = df[numeric_columns].describe()

            fig, axes = plt.subplots(3, 2, figsize=(20, 25))
            fig.suptitle("Miqdoriy o'zgaruvchilar uchun tavsifiy statistika", fontsize=16)

            for i, column in enumerate(numeric_columns):
                row = i // 2
                col = i % 2
                
                sns.boxplot(x=df[column], ax=axes[row, col])
                axes[row, col].set_title(f"{column} - Box Plot", fontsize=14)
                axes[row, col].set_xlabel(column, fontsize=12)
                
                # Statistik ma'lumotlarni qo'shamiz
                stats = desc_stats[column]
                textstr = '\n'.join((
                    f"Mean: {stats['mean']:.2f}",
                    f"Median: {stats['50%']:.2f}",
                    f"Std: {stats['std']:.2f}",
                    f"Min: {stats['min']:.2f}",
                    f"Max: {stats['max']:.2f}"
                ))
                
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                axes[row, col].text(0.95, 0.95, textstr, transform=axes[row, col].transAxes, 
                                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                                    bbox=props)

            plt.tight_layout()
            plt.subplots_adjust(top=0.95, hspace=0.3)
            st.pyplot(fig)

    with tab2:
            st.write('## Miqdoriy o\'zgaruvchilar korrelyatsiyasi')
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f')
            plt.title("Miqdoriy o'zgaruvchilar orasidagi korrelyatsiya", fontsize=14)
            st.pyplot(plt)

    with tab3:
            numeric_columns = ['carat', 'price', 'x', 'y', 'z', 'sold']
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(df[numeric_columns].isnull(), yticklabels=False, cbar=False, cmap='viridis')
            plt.title("Yo'q ma'lumotlar")
            st.pyplot(plt)
    with tab4:
        # Raqamli ustunlarni aniqlash
        numeric_columns = ['carat', 'price', 'x', 'y', 'z']
        categorical_columns = ['cut', 'color', 'clarity']

        # To'ldirishdan oldingi ma'lumotlarni saqlash
        df_original = df.copy()

        # Raqamli ustunlar uchun NaN qiymatlarni medianga almashtirish
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())

        # Kategorik ustunlar uchun NaN qiymatlarni mode bilan to'ldirish
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])

        st.title("NaN qiymatlarni to'ldirish natijasi")

        # Raqamli ustunlar uchun grafik
        fig_numeric, axes_numeric = plt.subplots(len(numeric_columns), 2, figsize=(15, 5 * len(numeric_columns)))
        fig_numeric.suptitle("Raqamli ustunlar: To'ldirishdan oldingi va keyingi taqsimotlar", fontsize=16, y=1)

        for i, column in enumerate(numeric_columns):
            # To'ldirishdan oldingi taqsimot
            sns.histplot(df_original[column].dropna(), ax=axes_numeric[i, 0], kde=True, color='blue')
            axes_numeric[i, 0].set_title(f"{column} - Original")
            axes_numeric[i, 0].set_xlabel("")

            # To'ldirishdan keyingi taqsimot
            sns.histplot(df[column], ax=axes_numeric[i, 1], kde=True, color='green')
            axes_numeric[i, 1].set_title(f"{column} - Median bilan to'ldirilgan")
            axes_numeric[i, 1].set_xlabel("")

        plt.tight_layout()
        st.pyplot(fig_numeric)

        # Kategorik ustunlar uchun grafik
        for column in categorical_columns:
            fig_cat, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            sns.countplot(data=df_original, x=column, ax=ax1, hue="cut")
            ax1.set_title(f"{column} - To'ldirishdan oldin")
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
            
            sns.countplot(data=df, x=column, ax=ax2, hue="cut")
            ax2.set_title(f"{column} - To'ldirishdan keyin")
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig_cat)
            
            # To'ldirilgan qiymatlar sonini ko'rsatish
            filled_count = df[column].count() - df_original[column].count()
            st.write(f"{column} ustuni uchun to'ldirilgan qiymatlar soni: {filled_count}")
            st.write("---")

        # Statistikalarni ko'rsatish
        col1, col2 = st.columns(2)

        with col1:
            st.write("To'ldirishdan oldingi statistika:")
            st.dataframe(df_original[numeric_columns].describe())

        with col2:
            st.write("To'ldirishdan keyingi statistika:")
            st.dataframe(df[numeric_columns].describe())
        with tab5:
                st.write('## Yillik sotuvlar va o\'rtacha narx dinamikasi')
                yearly_stats = df.groupby('years').agg({
                    'price': 'mean',
                    'sold': 'sum',
                    'carat': 'mean'
                })

                fig, ax1 = plt.subplots(figsize=(12, 6))
                ax2 = ax1.twinx()
                ax1.plot(yearly_stats.index, yearly_stats['sold'], 'g-')
                ax2.plot(yearly_stats.index, yearly_stats['price'], 'b-')
                ax1.set_xlabel('Yil')
                ax1.set_ylabel('Sotuvlar soni', color='g')
                ax2.set_ylabel("O'rtacha narx", color='b')
                
                
                
                plt.title("Yillik sotuvlar va o'rtacha narx dinamikasi")
                st.pyplot(fig)

    with tab6:
            # Outlierslarni aniqlash va olib tashlash
            Q1 = df['price'].quantile(0.25)
            Q3 = df['price'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_no_outliers = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

            # Grafiklarni chizish
            fig, axs = plt.subplots(2, 2, figsize=(20, 20))
            fig.suptitle('Olmoslar narxi taqsimoti', fontsize=24)

            # 1. Color bo'yicha narx taqsimoti (outlierlar bilan)
            sns.boxplot(x='color', y='price', data=df, ax=axs[0, 0], hue='color', legend=False)
            axs[0, 0].set_title('Color bo\'yicha narx (outlierlar bilan)', fontsize=18)
            axs[0, 0].set_xlabel('Rang', fontsize=14)
            axs[0, 0].set_ylabel('Narx', fontsize=14)

            # 2. Color bo'yicha narx taqsimoti (outlierlarsiz)
            sns.boxplot(x='color', y='price', data=df_no_outliers, ax=axs[0, 1], hue='color', legend=False)
            axs[0, 1].set_title('Color bo\'yicha narx (outlierlarsiz)', fontsize=18)
            axs[0, 1].set_xlabel('Rang', fontsize=14)
            axs[0, 1].set_ylabel('Narx', fontsize=14)

            # 3. Clarity bo'yicha narx taqsimoti (outlierlar bilan)
            sns.boxplot(x='clarity', y='price', data=df, ax=axs[1, 0], hue='clarity', legend=False)
            axs[1, 0].set_title('Clarity bo\'yicha narx (outlierlar bilan)', fontsize=18)
            axs[1, 0].set_xlabel('Tozalik', fontsize=14)
            axs[1, 0].set_ylabel('Narx', fontsize=14)

            # 4. Clarity bo'yicha narx taqsimoti (outlierlarsiz)
            sns.boxplot(x='clarity', y='price', data=df_no_outliers, ax=axs[1, 1], hue='clarity', legend=False)
            axs[1, 1].set_title('Clarity bo\'yicha narx (outlierlarsiz)', fontsize=18)
            axs[1, 1].set_xlabel('Tozalik', fontsize=14)
            axs[1, 1].set_ylabel('Narx', fontsize=14)

            # Grafiklarni chiroyli qilish
            for ax in axs.flat:
                ax.tick_params(axis='x')
                ax.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig)

    
    with tab8:
            st.write('## Kesim (cut) bo\'yicha narx taqsimoti')
            # Grafikni sozlash
            
            plt.figure(figsize=(12, 6))
            sns.set_style("whitegrid")

            # Boxplot va barplot kombinatsiyasini chizish
            ax = sns.boxplot(x='cut', y='price', data=df, hue='cut', legend=False)
            sns.barplot(x='cut', y='price', data=df, estimator=lambda x: x.median(), 
                        errorbar=None, alpha=0.3, hue='cut', legend=False)

            # Grafikni bezash
            plt.title("Olmosning kesimi (cut) bo'yicha narx taqsimoti", fontsize=16)
            plt.xlabel("Kesim (Cut)", fontsize=12)
            plt.ylabel("Narx", fontsize=12)

            # Median qiymatlarni ustunlar ustiga yozish
            medians = df.groupby('cut')['price'].median().round(2)
            for i, median in enumerate(medians):
                ax.text(i, median, f'${median}', ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            st.pyplot(plt)

            # Statistik ma'lumotlarni chiqarish
            st.write("Cut kategoriyalari bo'yicha narx statistikasi:")
            st.write(df.groupby('cut')['price'].describe())

            st.write("\nCut kategoriyalari bo'yicha olmoslar soni:")
            st.write(df['cut'].value_counts())

    
elif category == "Qisqa Tahlillar":
    
    st.subheader("")
    st.header("Statistik chizmalar")
    
    with st.expander("## Mamlakat bo'yicha olmoslar soni: "):
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Mamlakat bo'yicha olmoslar soni
        sns.countplot(data=df, x='country', ax=axes[0], hue="country")
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
        axes[0].set_title('Olmoslar Soni Mamlakat Bo\'yicha')
        axes[0].set_xlabel('Mamlakat')
        axes[0].set_ylabel('Son')

        # Mamlakatlarning ulushi
        df['country'].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[1])
        axes[1].set_title('Mamlakatlar Bo\'yicha Ulush')
        axes[1].set_ylabel('')

        plt.tight_layout()
        st.pyplot(plt)
        
    with st.expander("Karat bo'yicha taqsimot"):
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Karat bo'yicha taqsimot
        sns.histplot(df['carat'], bins=30, kde=True, ax=axes[0])
        axes[0].set_title('Karat Bo\'yicha Taqsimot')
        axes[0].set_xlabel('Karat')
        axes[0].set_ylabel('Son')

        # Karat bo'yicha box plot
        sns.boxplot(x=df['carat'], ax=axes[1])
        axes[1].set_title('Karat Bo\'yicha Box Plot')
        axes[1].set_xlabel('Karat')

        plt.tight_layout()
        st.pyplot(plt)
        
    with st.expander("Rang bo'yicha olmoslar soni"):
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Rang bo'yicha olmoslar soni
        sns.countplot(data=df, x='color', ax=axes[0], hue="color")
        axes[0].set_title('Olmoslar Rang Bo\'yicha')
        axes[0].set_xlabel('Rang')
        axes[0].set_ylabel('Son')

        # Ranglarning ulushi
        df['color'].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[1])
        axes[1].set_title('Ranglar Bo\'yicha Ulush')
        axes[1].set_ylabel('')

        plt.tight_layout()
        st.pyplot(plt)      
        
    with st.expander("Aniqlik bo'yicha olmoslar soni: "): 
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Anqlik bo'yicha olmoslar soni
        sns.countplot(data=df, x='clarity', ax=axes[0], hue="color")
        axes[0].set_title('Olmoslar Anqlik Bo\'yicha')
        axes[0].set_xlabel('Anqlik')
        axes[0].set_ylabel('Son')

        # Anqlik bo'yicha box plot
        sns.boxplot(x='clarity', y='carat', data=df, ax=axes[1])
        axes[1].set_title('Anqlik Bo\'yicha Karatning Taqsimoti')
        axes[1].set_xlabel('Anqlik')
        axes[1].set_ylabel('Karat')

        plt.tight_layout()
        st.pyplot(plt)      
        
        
    with st.expander("Mamlakat va Rang bo'yicha olmoslarning soni "):
        # Mamlakat va Rang bo'yicha olmoslarning soni
        pivot_country_color = df.pivot_table(index='country', columns='color', aggfunc='size', fill_value=0)
        print("Mamlakat va Rang bo'yicha Pivot Jadval:")
        print(pivot_country_color)

        # Pivot jadvalni chizish
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_country_color, cmap='viridis', annot=True, fmt='d')
        plt.title('Mamlakat va Rang Bo\'yicha Olmoslarning Sonlari')
        plt.xlabel('Rang')
        plt.ylabel('Mamlakat')
        st.pyplot(plt)
        
    with st.expander("Yil va Anqlik bo'yicha olmoslarning o'rtacha karat "):
        # Yil va Anqlik bo'yicha olmoslarning o'rtacha karat
        pivot_year_clarity = df.pivot_table(index='years', columns='clarity', values='carat', aggfunc='mean', fill_value=0)
        print("Yil va Anqlik bo'yicha Pivot Jadval:")
        print(pivot_year_clarity)

        # Pivot jadvalni chizish
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_year_clarity, cmap='viridis', annot=True, fmt='.2f')
        plt.title('Yil va Anqlik Bo\'yicha O\'rtacha Karat')
        plt.xlabel('Anqlik')
        plt.ylabel('Yil')
        st.pyplot(plt)
        
    with st.expander("Mamlakat, yil va o'rtacha narx bo'yicha pivot table "):
        

        # Mamlakat, yil va o'rtacha narx bo'yicha pivot table yaratish
        pivot_table = df.pivot_table(values='sold', index='country', columns='years', aggfunc='mean')

        # Pivot tableni heatmap yordamida ko'rsatish
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Mamlakatlar va Yillar bo\'yicha O\'rtacha Sotilgan Narx')
        plt.xlabel('Yil')
        plt.ylabel('Mamlakat')
        st.pyplot(plt)



   


        
        
elif category == "Xulosa":
    
    

    # Ma'lumotlarni yuklash
    @st.cache_data
    def load_data():
        df = pd.read_csv('16_1.csv')
        df = df.drop_duplicates()
        return df

    df = load_data()

    st.title("Qimmatbaho toshlar tahlili")

    # 1. Chiziqli grafik
    st.subheader("Narx trendi")
    fig, ax = plt.subplots(figsize=(10, 6))
    df.groupby('years')['price'].mean().plot(ax=ax)
    ax.set_xlabel("Yillar")
    ax.set_ylabel("O'rtacha narx")
    st.pyplot(fig)

    # 2. Doiraviy diagramma (Radial progress o'rniga)
    st.subheader("Kesim (cut) bo'yicha taqsimot")
    cut_counts = df['cut'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(cut_counts.values, labels=cut_counts.index, autopct='%1.1f%%')
    st.pyplot(fig)




    # 6. Sochma diagramma
    st.subheader("Karat va narx munosabati")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='carat', y='price', hue='cut', ax=ax)
    ax.set_xlabel("Karat")
    ax.set_ylabel("Narx")
    st.pyplot(fig)
    
    st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

    st.markdown('<p class="big-font">Muhim statistikalar</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("O'rtacha narx", f"${df['price'].mean():.2f}")
    col2.metric("Maksimal karat", f"{df['carat'].max():.2f}")
    #eng yuqori tozalikdagi olmos
    col3.metric("Noyob olmoslar soni", len(df[df['clarity'] == 'IF']))

    user_carat = st.slider('Karat', min_value=0.2, max_value=5.0, value=1.0)
    filtered_df = df[df['carat'] >= user_carat]


    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#f0f0f0')
        
    sns.scatterplot(data=filtered_df, x='carat', y='price', hue='cut', ax=ax)
        
    ax.set_title(f"{user_carat} karatdan yuqori olmoslar", fontsize=16, fontweight='bold')
    ax.set_xlabel("Karat", fontsize=12)
    ax.set_ylabel("Narx ($)", fontsize=12)
        
    ax.grid(True, linestyle='--', alpha=0.7)
        
    ax.legend(title='Kesim', bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.tight_layout()
        
    st.pyplot(fig)

    

    
    # Ma'lumotlarni CSV formatida yuklash
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df)

    st.sidebar.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_data.csv',
        mime='text/csv',
    )
    
    
    
    
