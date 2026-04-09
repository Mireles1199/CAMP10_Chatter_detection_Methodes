import os
import glob

docs_path = r"d:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\Chatter-Criteria\CAMP10_Chatter_detection_Methodes\MaxEnt_SPRT\MaxEnt_SPRT\docs"

md_files = glob.glob(os.path.join(docs_path, "**", "*.md"), recursive=True)

print(f"Encontrados {len(md_files)} archivos .md para reparar\n")

for file_path in md_files:
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        
        # Use byte escape sequences
        original_content = content
        
        content = content.replace('\xe2\x80\x94', chr(0x2014))
        content = content.replace('\xe2\x80\x93', chr(0x2013))
        content = content.replace('\xc3\x97', chr(0x00D7))
        content = content.replace('\xe2\x86\x92', chr(0x2192))
        content = content.replace('\xe2\x89\xa5', chr(0x2265))
        content = content.replace('\xe2\x80\x9c', chr(0x201C))
        content = content.replace('\xe2\x80\x9d', chr(0x201D))
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"OK: {os.path.basename(file_path)}")
        else:
            print(f"OK: {os.path.basename(file_path)}")
            
    except Exception as e:
        print(f"Error: {os.path.basename(file_path)}: {e}")

print("\nDone processing markdown files")

