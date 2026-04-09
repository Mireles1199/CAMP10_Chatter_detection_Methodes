import os
import glob

docs_path = r"d:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Chatter-Criteria\CAMP10_Chatter_detection_Methodes\MaxEnt_SPRT\MaxEnt_SPRT\docs"

md_files = glob.glob(os.path.join(docs_path, "**", "*.md"), recursive=True)

print(f"Proces {len(md_files)} archivos\n")

for file_path in md_files:
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        
        original = data
        
        # The actual malformed sequences found in files
        # c3 a2 e2 82 ac e2 80 9d = corrupted em-dash
        data = data.replace(b'\xc3\xa2\xe2\x82\xac\xe2\x80\x9d', '—'.encode('utf-8'))
        # c3 a2 e2 82 ac = corrupted prefix
        data = data.replace(b'\xc3\xa2\xe2\x82\xac', '–'.encode('utf-8'))
        # c3 a2 e2 80 a0 e2 80 99 = corrupted arrow
        data = data.replace(b'\xc3\xa2\xe2\x80\xa0\xe2\x80\x99', '→'.encode('utf-8'))
        # c3 a2 e2 86 92 = corrupted arrow alternate
        data = data.replace(b'\xc3\xa2\xe2\x86\x92', '→'.encode('utf-8'))
        # c3 a2 e2 89 a5 = corrupted greater equal
        data = data.replace(b'\xc3\xa2\xe2\x89\xa5', '≥'.encode('utf-8'))
        # c3 83 c2 97 = corrupted times
        data = data.replace(b'\xc3\x83\xc2\x97', '×'.encode('utf-8'))
        
        if data != original:
            with open(file_path, 'wb') as f:
                f.write(data)
            print(f"Fixed: {os.path.basename(file_path)}")
        else:
            print(f"OK: {os.path.basename(file_path)}")
            
    except Exception as e:
        print(f"Error {os.path.basename(file_path)}: {e}")

print("\nAll files processed")

