import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Konfigurasi Path data
RAW_DATA_PATH = 'titanic_raw/titanic.csv'
OUTPUT_PATH = 'preprocessing/titanic_preprocessing.csv'

# Helper Functions Data Preprocessing
def load_data(path):
    """
    Fungsi untuk memuat data dari file CSV.
    """
    print(f"Loading data dari {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan di: {path}")
    
    df = pd.read_csv(path)
    print(f"Data berhasil dimuat. Ukuran: {df.shape}")
    return df

# Fungsi untuk mengatasi missing value
def handle_missing_values(df):
    """
    Fungsi untuk menangani missing values spesifik untuk dataset Titanic.
    
    Strategi Update:
    1. Age: Diisi dengan Median.
    2. Embarked: Diisi dengan Modus.
    3. Cabin: DIUBAH menjadi fitur 'HasCabin' (1 jika ada, 0 jika kosong), 
       lalu kolom asli 'Cabin' dihapus.
    """
    df = df.copy()
    print("Memulai Handling Missing Values...")

    # Input missing value kolom Age dengan Median 
    # karena terdapat outlier dan distribusi sedikit ada kemiringan 
    if 'Age' in df.columns:
        median_age = df['Age'].median()
        df['Age'] = df['Age'].fillna(median_age)
        
    # Input missing value kolom Embarked dengan Mode/Modus
    if 'Embarked' in df.columns:
        mode_embarked = df['Embarked'].mode()[0]
        df['Embarked'] = df['Embarked'].fillna(mode_embarked)
        
    # Mengatasi missing value pada kolom Cabin 
    # dengan membuat Kolom hasCabin 
    if 'Cabin' in df.columns:
        # Buat kolom baru (isi 1 jika Cabin ada isinya, 0 jika NaN)
        df['hasCabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
        # Hapus kolom asli Cabin karena sudah diwakili HasCabin
        df = df.drop(columns=['Cabin'])
        
    # Bersihkan sisa-sisa missing value jika ada
    if df.isnull().sum().sum() > 0:
        print("Masih ada sisa missing values drop rows...")
        df.dropna(inplace=True)
    
    print("Selesai Handling Missing Values.\n")
    return df

# Fungsi untuk melakukan teknik rekayasa fitur (feature engineering)
def feature_engineering(df):
    """
    Fungsi untuk melakukan Feature Engineering dan Feature Selection.
    
    Langkah-langkah:
    1. Extract Title: Mengambil gelar (Mr, Mrs, dll) dari kolom Name.
    2. FamilySize: Menggabungkan SibSp dan Parch.
    3. Drop Columns: Menghapus kolom yang tidak lagi relevan atau redundan.
    """
    df = df.copy()
    print("Memulai Feature Engineering...")

    # Membuat Fitur 'Title' dari kolom 'Name'
    if 'Name' in df.columns:
        # Menggunakan RegEx untuk mengambil kata yang diakhiri titik (.)
        # Contoh: "Braund, Mr. Owen Harris", diambil "Mr"
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        
        # Mengelompokkan gelar-gelar langka agar model tidak rancu
        title_mapping = {
            'Mr': 'Mr', 
            'Miss': 'Miss', 'Mlle': 'Miss', 'Ms': 'Miss', # Penyeragaman Miss
            'Mrs': 'Mrs', 'Mme': 'Mrs',                   # Penyeragaman Mrs
            'Master': 'Master',
            # Gelar bangsawan/profesi langka digabung jadi 'Rare'
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare', 
            'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare', 
            'Don': 'Rare', 'Dona': 'Rare', 'Capt': 'Rare', 'Sir': 'Rare'
        }
        # Isi dengan 'Rare' jika ada gelar yang tidak terdaftar di mapping
        df['Title'] = df['Title'].map(title_mapping).fillna('Rare')

    # Membuat Fitur 'FamilySize' berdasarkan kolom SibSp dan Parch
    # Logic: Saudara/Pasangan + Orangtua/Anak + Diri Sendiri atau 1
    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
    # Feature Selection (Drop Kolom)
    # Kolom yang akan dibuang:
    # - PassengerId: Hanya index urut, tidak ada pola.
    # - Name: Sudah diambil melalui 'Title'.
    # - Ticket: Terlalu acak/unik, susah ditarik polanya.
    cols_to_drop = ['PassengerId', 'Name', 'Ticket']
    
    # Cek dulu apakah kolomnya ada sebelum di-drop (biar tidak error saat re-run)
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
        
    print("Selesai Feature Engineering.\n")
    return df

# Fungsi untuk melakukan feature encoding
def encoding(df):
    """
    Fungsi untuk mengubah data kategori (Teks) menjadi Angka.
    
    Target Kolom:
    - Sex (male, female) -> 0, 1
    - Embarked (S, C, Q) -> 0, 1, 2
    - Title (Mr, Mrs, dll) -> 0, 1, 2, ...
    """
    df = df.copy()
    print("Memulai Encoding Data...")

    # Ambil semua kolom yang tipe datanya 'object' (Teks)
    # Ini otomatis akan mengambil Sex, Embarked, dan Title
    cat_cols = df.select_dtypes(include=['object']).columns    
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    print("Selesai Encoding.\n")
    return df

# Fungsi untuk melakukan feature scaling
def scaling(df):
    """
    Fungsi untuk melakukan penyekalaan (Scaling) pada fitur numerik.
    
    Teknik:
    1. Log Transformation: Khusus untuk 'Fare' yang skewed (miring).
       Menggunakan np.log1p (log(1+x)) agar tidak error jika ada nilai 0.
    2. StandardScaler: Mengubah distribusi menjadi Mean=0, Std=1.
    
    Target Kolom: Age, Fare
    """
    df = df.copy()
    print("Memulai Scaling Data...")

    # Melakukan Log Transform untuk 'Fare'
    # Ini akan mengubah grafik yang miring tajam menjadi lebih seperti lonceng (normal)
    if 'Fare' in df.columns:
        # Menggunakan log1p (log(1+x)) karena Fare ada yang nilainya 0
        df['Fare'] = np.log1p(df['Fare'])
        
    # Melakukan Standard Scaling
    # Target kolom numerik
    num_cols = ['Age', 'Fare']
    
    # Cek apakah kolomnya ada di df
    valid_cols = [c for c in num_cols if c in df.columns]
    
    if valid_cols:
        scaler = StandardScaler()
        df[valid_cols] = scaler.fit_transform(df[valid_cols])
        
    print("Selesai Scaling.\n")
    return df

def save_data(df, path):
    """
    Fungsi untuk menyimpan data bersih ke file CSV.
    """
    print(f"Menyimpan data ke {path}...")
    
    # Pastikan direktori ada (jika belum ada)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    df.to_csv(path, index=False)
    print("Data berhasil disimpan! Proses Otomatisasi Selesai.")

# Main Pipeline Function
def main():
    try:
        # Pipeline Eksekusi
        df = load_data(RAW_DATA_PATH)
        df = handle_missing_values(df)
        df = feature_engineering(df)
        df = encoding(df)
        df = scaling(df)
        save_data(df, OUTPUT_PATH)
        
    except Exception as e:
        print(f"\n[ERROR] Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()