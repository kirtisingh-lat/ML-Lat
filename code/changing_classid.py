from pathlib import Path

LABELS_DIR = Path("/home/ss/Kirti/lat/datasets/people_only/labels2/test")  # change to your labels root
OLD_ID = 5
NEW_ID = 0

# Check if the directory exists and print files found
print(f"Looking for .txt files in: {LABELS_DIR}")
for txt in LABELS_DIR.rglob("*.txt"):
    #print(f"Processing file: {txt}")
    
    out_lines = []
    for line in txt.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls = int(parts[0])
        
        # Print the current class and check the condition
        #print(f"Original Class: {cls}")
        
        if cls == OLD_ID:
            parts[0] = str(NEW_ID)
        
        out_lines.append(" ".join(parts))
    
    # Write the updated content back to the file
    txt.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    print(f"Finished processing {txt}")
