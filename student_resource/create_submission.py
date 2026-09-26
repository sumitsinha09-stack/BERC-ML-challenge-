import os
import sys
import zipfile
import subprocess

def make_submission(team_name="BERC_Team"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    zip_filename = os.path.join(base_dir, f"{team_name}_submission.zip")
    
    print(f"=== Creating Submission Package: {zip_filename} ===")
    
    # 1. Run Validator first
    print("\n--- Running Official Submission Validator ---")
    val_script = os.path.join(base_dir, "utils", "validate_submission.py")
    matching_file = os.path.join(base_dir, "output", "matching_results.tsv")
    candidate_file = os.path.join(base_dir, "output", "candidate_pairs.tsv")
    test_dir = os.path.join(base_dir, "dataset", "test")
    
    cmd = [
        sys.executable, val_script,
        "--matching", matching_file,
        "--candidate", candidate_file,
        "--test-dir", test_dir
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("Validation FAILED! Please fix the errors before creating zip.")
        if res.stderr:
            print("STDERR:", res.stderr)
        return False
        
    print("\n--- Packaging Files into ZIP ---")
    files_to_pack = [
        # Output files
        ("output/matching_results.tsv", "output/matching_results.tsv"),
        ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        # Documentation
        ("Documentation_template.md", "Documentation_template.md"),
        # Code
        ("code/business_entity_resolution/README.md", "code/business_entity_resolution/README.md"),
        ("code/business_entity_resolution/requirements.txt", "code/business_entity_resolution/requirements.txt"),
    ]
    
    # Add all src files
    src_dir = os.path.join(base_dir, "code", "business_entity_resolution", "src")
    for f in os.listdir(src_dir):
        if f.endswith(".py") or f.endswith(".pkl") or f.endswith(".txt"):
            rel_path = os.path.join("code", "business_entity_resolution", "src", f)
            files_to_pack.append((rel_path, rel_path))
            
    with zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for local_rel, zip_rel in files_to_pack:
            local_full = os.path.join(base_dir, local_rel)
            if os.path.exists(local_full):
                zf.write(local_full, arcname=zip_rel)
                print(f"  Added: {zip_rel} ({os.path.getsize(local_full)/1024:.1f} KB)")
            else:
                print(f"  WARNING: File missing: {local_full}")
                
    print(f"\nSUCCESS! Final submission package created at: {zip_filename}")
    print(f"Package Size: {os.path.getsize(zip_filename) / (1024*1024):.2f} MB")
    
    # Also place copies in workspace root for easy user submission
    root_dir = os.path.dirname(base_dir)
    root_zip = os.path.join(root_dir, f"{team_name}_submission.zip")
    root_match = os.path.join(root_dir, "matching_results.tsv")
    import shutil
    shutil.copy2(zip_filename, root_zip)
    shutil.copy2(os.path.join(base_dir, "output", "matching_results.tsv"), root_match)
    print(f"Placed direct submission copies at:\n  1. {root_zip}\n  2. {root_match}")
    return True

if __name__ == '__main__':
    team = sys.argv[1] if len(sys.argv) > 1 else "BERC_Team"
    make_submission(team)
