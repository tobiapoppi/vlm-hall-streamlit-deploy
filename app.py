import streamlit as st
import json
import os
import uuid
from datetime import datetime
import pandas as pd
import random
import re
import tempfile
import shutil
import zipfile
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Data Evaluation App",
    page_icon="📊",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MEDIA_KEYS = {"video", "image", "chosen_video", "rejected_video", "chosen_image", "rejected_image"}

ORIGINAL_FILE = str(DATA_DIR / "val_mixed_dpo.jsonl")
LABELED_FILE = str(DATA_DIR / "val_mixed_dpo_human_labelled_v2___template.jsonl")
PROGRESS_FILE = str(DATA_DIR / "evaluation_progress_v2.json")
OLD_LABELED_FILE = str(DATA_DIR / "val_mixed_dpo_human_labelled_v2.jsonl")


def resolve_local_path(path_value):
    """Resolve repository-relative media paths to absolute paths."""
    if not isinstance(path_value, str) or not path_value:
        return path_value
    candidate = Path(path_value)
    if candidate.is_absolute():
        return path_value
    return str((APP_DIR / candidate).resolve())


def load_jsonl(file_path):
    """Load JSONL file and return list of dictionaries"""
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    example = json.loads(line)
                    for key in MEDIA_KEYS:
                        if key in example:
                            example[key] = resolve_local_path(example[key])
                    data.append(example)
    return data


def save_to_jsonl(data, file_path):
    """Append data to JSONL file"""
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data) + '\n')


def load_progress():
    """Load evaluation progress"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            data = json.load(f)
        
        # Migrate old format to new format
        if "user_aliases" not in data:
            data["user_aliases"] = {}
        if "alias_stats" not in data:
            data["alias_stats"] = {}
        if "assigned_videos" not in data:
            data["assigned_videos"] = {"ACTREC": {}, "ACTSEQ": {}}
        
        # Save migrated data
        save_progress(data)
        return data
    
    return {
        "evaluated_ids": [], 
        "user_sessions": {},
        "user_aliases": {},
        "alias_stats": {},
        "assigned_videos": {"ACTREC": {}, "ACTSEQ": {}}
    }


def save_progress(progress_data):
    """Save evaluation progress"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress_data, f, indent=2)


def get_user_session_id():
    """Get or create user session ID"""
    if 'user_session_id' not in st.session_state:
        st.session_state.user_session_id = str(uuid.uuid4())
    return st.session_state.user_session_id


def get_user_alias():
    """Get user alias from session state"""
    return st.session_state.get('user_alias', None)


def set_user_alias(alias, progress_data):
    """Set user alias and update progress data"""
    session_id = get_user_session_id()
    st.session_state.user_alias = alias
    
    # Update progress data structures
    if alias not in progress_data["user_aliases"]:
        progress_data["user_aliases"][alias] = []
    
    if session_id not in progress_data["user_aliases"][alias]:
        progress_data["user_aliases"][alias].append(session_id)
    
    # Initialize session data if not exists
    if session_id not in progress_data["user_sessions"]:
        progress_data["user_sessions"][session_id] = {
            "alias": alias,
            "position": 0,
            "labeled_count": 0
        }
    else:
        progress_data["user_sessions"][session_id]["alias"] = alias
    
    # Initialize alias stats if not exists
    if alias not in progress_data["alias_stats"]:
        progress_data["alias_stats"][alias] = {
            "total_labeled": 0,
            "sessions": []
        }
    
    if session_id not in progress_data["alias_stats"][alias]["sessions"]:
        progress_data["alias_stats"][alias]["sessions"].append(session_id)
    
    save_progress(progress_data)


def show_alias_input():
    """Show alias input form"""
    st.title("👋 Welcome to Data Evaluation App")
    st.markdown("---")
    
    st.subheader("Please enter your alias to continue")
    st.write("This helps us track your contributions and avoid duplicate work.")
    
    with st.form("alias_form"):
        alias = st.text_input(
            "Your Alias:",
            placeholder="e.g., tobipop, floschi, ...",
            help="You can use the same alias across different devices/sessions"
        )
        submitted = st.form_submit_button("Continue", type="primary")
        
        if submitted:
            if alias.strip():
                return alias.strip()
            else:
                st.error("Please enter a valid alias")
    
    return None


def get_user_stats(alias, progress_data):
    """Get statistics for a specific user alias"""
    if alias not in progress_data["alias_stats"]:
        return {"total_labeled": 0, "sessions": []}
    
    # Count actual labeled examples by this alias
    labeled_data = load_jsonl(LABELED_FILE)
    user_labeled_count = sum(1 for item in labeled_data if item.get("evaluator_alias") == alias)
    
    return {
        "total_labeled": user_labeled_count,
        "sessions": progress_data["alias_stats"][alias]["sessions"]
    }


def get_active_users_count(progress_data):
    """Get count of active users (users with at least one session)"""
    return len(progress_data["user_aliases"])


def parse_sample_id(sample_id):
    """
    Parse sample ID to extract task family, variant, preference type, and video ID
    
    Examples:
    - PVDpairACTREC_A_out_004040_01 -> ("ACTREC", "A", "out", "004040")
    - PVDpairACTREC_A_in_004040_01 -> ("ACTREC", "A", "in", "004040")
    - PVDpair_C_out_004040_01 -> ("ACTSEQ", "C", "out", "004040")
    """
    # Determine task family
    if "ACTREC" in sample_id:
        task_family = "ACTREC"
    else:
        task_family = "ACTSEQ"
    
    # Extract task variant (A, B, C, or D)
    # Pattern: look for _<letter>_ in the ID
    import re
    variant_match = re.search(r'_([A-D])_', sample_id)
    task_variant = variant_match.group(1) if variant_match else "UNKNOWN"
    
    # Extract preference type (in or out)
    if '_in_' in sample_id:
        preference_type = "in"
        video_match = re.search(r'in_(\d+)', sample_id)
    elif '_out_' in sample_id:
        preference_type = "out"
        video_match = re.search(r'out_(\d+)', sample_id)
    else:
        preference_type = "UNKNOWN"
        video_match = None
    
    # Extract video ID
    video_id = video_match.group(1) if video_match else "UNKNOWN"
    
    return task_family, task_variant, preference_type, video_id


def load_old_evaluated_ids():
    """
    Load IDs from old labeled file and convert them to new format.
    Old IDs need 'PVD' prefix to match new ID format.
    """
    old_evaluated_ids = set()
    
    if os.path.exists(OLD_LABELED_FILE):
        old_labeled_data = load_jsonl(OLD_LABELED_FILE)
        for item in old_labeled_data:
            old_id = item.get("id", "")
            # Add 'PVD' prefix to make it compatible with new IDs
            if old_id and not old_id.startswith("PVD"):
                new_format_id = "PVD" + old_id
                old_evaluated_ids.add(new_format_id)
            elif old_id:
                old_evaluated_ids.add(old_id)
    
    return old_evaluated_ids


def group_samples_by_task(samples, exclude_ids=None):
    """
    Group samples by task type and preference type (e.g., ACTREC_A_in, ACTREC_A_out)
    
    Args:
        samples: List of sample dictionaries
        exclude_ids: Set of IDs to exclude from grouping
    """
    if exclude_ids is None:
        exclude_ids = set()
    
    grouped = {
        "ACTREC_A_in": [],
        "ACTREC_A_out": [],
        "ACTREC_C_in": [],
        "ACTREC_C_out": [],
        "ACTREC_D_in": [],
        "ACTREC_D_out": [],
        "ACTSEQ_A_in": [],
        "ACTSEQ_A_out": [],
        "ACTSEQ_B_in": [],
        "ACTSEQ_B_out": [],
        "ACTSEQ_C_in": [],
        "ACTSEQ_C_out": []
    }
    
    for sample in samples:
        # Skip if this sample was already evaluated in old study
        if sample["id"] in exclude_ids:
            continue
            
        task_family, task_variant, preference_type, video_id = parse_sample_id(sample["id"])
        task_key = f"{task_family}_{task_variant}_{preference_type}"
        
        if task_key in grouped:
            grouped[task_key].append({
                "sample": sample,
                "video_id": video_id,
                "task_family": task_family,
                "task_variant": task_variant,
                "preference_type": preference_type
            })
    
    return grouped


def generate_user_assignment(grouped_samples, already_assigned_videos, session_id):
    """
    Generate a 30-sample assignment for a user with balanced input/output preferences:
    - 3 ACTREC_C_in, 3 ACTREC_C_out
    - 3 ACTREC_D_in, 3 ACTREC_D_out  
    - 3 ACTREC_A_in, 3 ACTREC_A_out (18 total ACTREC, unique video IDs within ACTREC)
    - 2 ACTSEQ_C_in, 2 ACTSEQ_C_out
    - 2 ACTSEQ_B_in, 2 ACTSEQ_B_out
    - 2 ACTSEQ_A_in, 2 ACTSEQ_A_out (12 total ACTSEQ, unique video IDs within ACTSEQ)
    - Phase 1: all "_out" samples first (single-video examples)
    - Phase 2: all "_in" samples second (single-answer examples)
    """
    assignment = []
    assigned_video_ids = {
        "ACTREC": set(),
        "ACTSEQ": set()
    }
    
    # Define task order split in two phases:
    # Phase 1: single-video examples ("out")
    # Phase 2: single-answer examples ("in")
    task_order = [
        ("ACTREC_C_out", 3),
        ("ACTREC_D_out", 3),
        ("ACTREC_A_out", 3),
        ("ACTSEQ_C_out", 2),
        ("ACTSEQ_B_out", 2),
        ("ACTSEQ_A_out", 2),
        ("ACTREC_C_in", 3),
        ("ACTREC_D_in", 3),
        ("ACTREC_A_in", 3),
        ("ACTSEQ_C_in", 2),
        ("ACTSEQ_B_in", 2),
        ("ACTSEQ_A_in", 2)
    ]
    
    # Use session_id as seed for reproducible but unique assignments per user
    rng = random.Random(session_id)
    
    for task_key, count in task_order:
        task_family = task_key.split("_")[0]
        available_samples = grouped_samples[task_key].copy()
        
        # Shuffle available samples for this task
        rng.shuffle(available_samples)
        
        selected_count = 0
        for sample_info in available_samples:
            video_id = sample_info["video_id"]
            
            # Check constraints
            if video_id in assigned_video_ids[task_family]:
                continue  # Skip if video already used in this task family
            
            if video_id in already_assigned_videos.get(task_family, {}):
                continue  # Skip if video already assigned to another user
            
            # Add to assignment
            assignment.append(sample_info["sample"]["id"])
            assigned_video_ids[task_family].add(video_id)
            selected_count += 1
            
            if selected_count >= count:
                break
        
        # Check if we got enough samples
        if selected_count < count:
            raise Exception(f"Not enough unique samples available for {task_key}. Got {selected_count}, needed {count}")
    
    return assignment, assigned_video_ids


def get_or_create_assignment(original_data, progress_data, session_id):
    """Get or create assignment for a user session"""
    session_data = progress_data["user_sessions"].get(session_id)
    
    # Check if assignment already exists
    if session_data and isinstance(session_data, dict) and "assignment" in session_data:
        return session_data["assignment"]
    
    # Generate new assignment
    st.info("🔄 Generating your personalized assignment...")
    
    try:
        # Load old evaluated IDs to exclude them
        old_evaluated_ids = load_old_evaluated_ids()
        
        # Group samples by task type, excluding old evaluated IDs
        grouped_samples = group_samples_by_task(original_data, exclude_ids=old_evaluated_ids)
        
        # Get already assigned videos
        already_assigned = progress_data.get("assigned_videos", {"ACTREC": {}, "ACTSEQ": {}})
        
        # Generate assignment
        assignment, assigned_video_ids = generate_user_assignment(
            grouped_samples, 
            already_assigned, 
            session_id
        )
        
        # Update progress data with assignment
        if session_id not in progress_data["user_sessions"]:
            progress_data["user_sessions"][session_id] = {}
        
        progress_data["user_sessions"][session_id]["assignment"] = assignment
        progress_data["user_sessions"][session_id]["position"] = 0
        progress_data["user_sessions"][session_id]["labeled_count"] = 0
        
        # Update assigned videos tracking
        for task_family in ["ACTREC", "ACTSEQ"]:
            for video_id in assigned_video_ids[task_family]:
                if video_id not in progress_data["assigned_videos"][task_family]:
                    progress_data["assigned_videos"][task_family][video_id] = []
                progress_data["assigned_videos"][task_family][video_id].append(session_id)
        
        save_progress(progress_data)
        st.success("✅ Assignment generated successfully!")
        
        return assignment
        
    except Exception as e:
        st.error(f"Error generating assignment: {str(e)}")
        return None


def get_next_example(original_data, progress_data, session_id):
    """Get next example from user's assignment"""
    # Get or create assignment
    assignment = get_or_create_assignment(original_data, progress_data, session_id)
    
    if not assignment:
        return None, -1
    
    # Get current position
    session_data = progress_data["user_sessions"].get(session_id, {})
    position = session_data.get("position", 0)
    
    # Check if we've completed all assigned samples
    if position >= len(assignment):
        return None, -1
    
    # Get the sample ID at current position
    sample_id = assignment[position]
    
    # Find the sample in original data
    for idx, sample in enumerate(original_data):
        if sample["id"] == sample_id:
            return sample, position
    
    # If sample not found, skip to next
    progress_data["user_sessions"][session_id]["position"] = position + 1
    save_progress(progress_data)
    return get_next_example(original_data, progress_data, session_id)


def prepare_sample_download(example):
    """Prepare a ZIP file containing all data for the current sample"""
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        sample_id = example.get("id", "unknown")
        sample_folder = os.path.join(temp_dir, f"sample_{sample_id}")
        os.makedirs(sample_folder, exist_ok=True)
        
        # Collect media files to copy
        media_files = []
        swap_type = example.get("swap_type", "")
        modality = example.get("modality", "")
        
        if swap_type == "input_swap":
            if modality == "video":
                if "chosen_video" in example and os.path.exists(example["chosen_video"]):
                    media_files.append(("chosen_video", example["chosen_video"]))
                if "rejected_video" in example and os.path.exists(example["rejected_video"]):
                    media_files.append(("rejected_video", example["rejected_video"]))
            elif modality == "image":
                if "chosen_image" in example and os.path.exists(example["chosen_image"]):
                    media_files.append(("chosen_image", example["chosen_image"]))
                if "rejected_image" in example and os.path.exists(example["rejected_image"]):
                    media_files.append(("rejected_image", example["rejected_image"]))
        
        elif swap_type == "output_swap":
            if modality == "video":
                if "video" in example and os.path.exists(example["video"]):
                    media_files.append(("video", example["video"]))
            elif modality == "image":
                if "image" in example and os.path.exists(example["image"]):
                    media_files.append(("image", example["image"]))
        
        # Copy media files to sample folder
        copied_files = {}
        for file_key, file_path in media_files:
            try:
                file_name = os.path.basename(file_path)
                dest_path = os.path.join(sample_folder, file_name)
                shutil.copy2(file_path, dest_path)
                copied_files[file_key] = file_name
            except Exception as e:
                copied_files[file_key] = f"ERROR: {str(e)}"
        
        # Create metadata JSON with all text data
        metadata = {
            "id": example.get("id"),
            "task_type": example.get("task_type"),
            "modality": modality,
            "swap_type": swap_type,
            "prompt": example.get("prompt"),
            "media_files": copied_files
        }
        
        # Add answers based on swap type
        if swap_type == "input_swap":
            metadata["answer"] = example.get("answer")
        elif swap_type == "output_swap":
            metadata["chosen_answer"] = example.get("chosen_answer")
            metadata["rejected_answer"] = example.get("rejected_answer")
        
        # Add any additional metadata fields
        for key in ["dataset_source", "video_id", "image_id"]:
            if key in example:
                metadata[key] = example[key]
        
        # Save metadata to JSON file
        metadata_path = os.path.join(sample_folder, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Create ZIP file
        zip_path = os.path.join(temp_dir, f"sample_{sample_id}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files from sample folder
            for root, dirs, files in os.walk(sample_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join(f"sample_{sample_id}", os.path.relpath(file_path, sample_folder))
                    zipf.write(file_path, arcname)
        
        # Read ZIP file into memory
        with open(zip_path, 'rb') as f:
            zip_data = f.read()
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        return zip_data, f"sample_{sample_id}.zip"
    
    except Exception as e:
        # Clean up on error
        if 'temp_dir' in locals():
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        raise Exception(f"Error preparing download: {str(e)}")


def display_media_content(example):
    """Display media content based on swap type and modality"""
    swap_type = example.get("swap_type", "")
    modality = example.get("modality", "")

    st.write(f"**Task:** {example.get('task_type', 'Unknown')} | **Modality:** {modality.title()} | **Swap Type:** {swap_type.replace('_', ' ').title()}")

    if swap_type == "input_swap":
        # Show comparison between chosen and rejected inputs
        if modality == "video":
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Chosen Video:**")
                if os.path.exists(example["chosen_video"]):
                    st.video(example["chosen_video"])
                else:
                    st.error(f"Video not found: {example['chosen_video']}")

            with col2:
                st.write("**Rejected Video:**")
                if os.path.exists(example["rejected_video"]):
                    st.video(example["rejected_video"])
                else:
                    st.error(f"Video not found: {example['rejected_video']}")

        elif modality == "image":
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Chosen Image:**")
                if os.path.exists(example["chosen_image"]):
                    st.image(example["chosen_image"], use_container_width=True)
                else:
                    st.error(f"Image not found: {example['chosen_image']}")

            with col2:
                st.write("**Rejected Image:**")
                if os.path.exists(example["rejected_image"]):
                    st.image(example["rejected_image"], use_container_width=True)
                else:
                    st.error(f"Image not found: {example['rejected_image']}")

    elif swap_type == "output_swap":
        # Show single input with different answers - use column layout for consistent sizing
        col1, col2 = st.columns(2)
        with col1:
            if modality == "video":
                st.write("**Video (Reference):**")
                if os.path.exists(example["video"]):
                    st.video(example["video"])
                else:
                    st.error(f"Video not found: {example['video']}")

            elif modality == "image":
                st.write("**Image (Reference):**")
                if os.path.exists(example["image"]):
                    st.image(example["image"], use_container_width=True)
                else:
                    st.error(f"Image not found: {example['image']}")
        # Leave col2 empty to maintain consistent layout


def format_text_with_actions(text):
    """Format text to make actions more visible with numbered bold formatting"""
    if not text:
        return text
    
    # Look for "Here are the possible actions:" pattern and format the numbered list
    if re.search(r'Here are the possible actions?:', text, re.IGNORECASE):
        # Find the pattern: "Given actions: 1 Action; 2 Action; etc."
        pattern = r'(.*?)(Here are the possible actions?:)\s*(.*?)(\s*Which number corresponds to the video\?.*)'
        match = re.match(pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            before = match.group(1)
            given_actions_text = match.group(2)
            actions_text = match.group(3)
            after = match.group(4)
            
            # Split by semicolons and format each action
            if ';' in actions_text:
                actions = [action.strip() for action in actions_text.split(';') if action.strip()]
                formatted_actions = []
                
                for action in actions:
                    # Get the original number FIRST, before any processing
                    num_match = re.match(r'^(\d+)\.\s*(.*)$', action.strip())
                    if num_match:
                        num = num_match.group(1)
                        action_text = num_match.group(2).rstrip('.')  # Remove trailing period
                        formatted_actions.append(f"**{num}.** {action_text}")
                    else:
                        # Fallback for actions without numbering
                        clean_action = action.strip().rstrip('.')
                        if clean_action:
                            formatted_actions.append(clean_action)
                
                # Add line break before "Provide the correct order" instruction
                if re.search(r'Which number corresponds to the video', after):
                    after = "\n\n" + after.strip()
                
                # Reconstruct the text - use plain text for "Given actions:" since bold isn't working
                formatted_text = before + f"{given_actions_text}\n\n" + ";\n\n".join(formatted_actions) + "." + after
                return formatted_text

    # Look for "Given actions:" pattern and format the numbered list
    if re.search(r'given actions?:', text, re.IGNORECASE):
        # Find the pattern: "Given actions: 1 Action; 2 Action; etc."
        pattern = r'(.*?)(given actions?:)\s*([^.]*\.)(.*)' 
        match = re.match(pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            before = match.group(1)
            given_actions_text = match.group(2)
            actions_text = match.group(3)
            after = match.group(4)
            
            # Split by semicolons and format each action
            if ';' in actions_text:
                actions = [action.strip() for action in actions_text.split(';') if action.strip()]
                formatted_actions = []
                
                for action in actions:
                    # Remove the period at the end if it's the last action
                    action = action.rstrip('.')
                    # Remove existing numbering and format with bold
                    clean_action = re.sub(r'^\d+\s*', '', action).strip()
                    if clean_action:
                        # Get the original number if it exists
                        num_match = re.match(r'^(\d+)\s*(.*)$', action)
                        if num_match:
                            num = num_match.group(1)
                            action_text = num_match.group(2)
                            formatted_actions.append(f"**{num}.** {action_text}")
                        else:
                            formatted_actions.append(clean_action)
                
                # Add line break before final question and make the sequence in brackets bold
                if re.search(r'Is\s*\[[^\]]+\]', after):
                    after = "\n\n" + after.strip()
                    # Make the sequence in square brackets bold
                    after = re.sub(r'\[([^\]]+)\]', r'**[\1]**', after)
                
                # Add line break before "Provide the correct order" instruction
                if re.search(r'Provide the correct order as comma-separated indices', after):
                    after = "\n\n" + after.strip()
                
                # Reconstruct the text - use plain text for "Given actions:" since bold isn't working
                formatted_text = before + f"{given_actions_text}\n\n" + ";\n\n".join(formatted_actions) + "." + after
                return formatted_text
    
    return text


def display_text_content(example):
    """Display prompt and answer(s)"""
    # Display prompt
    st.markdown("**Prompt:**")
    formatted_prompt = format_text_with_actions(example["prompt"])
    st.markdown(formatted_prompt)

    swap_type = example.get("swap_type", "")

    if swap_type == "input_swap":
        # Single answer for both inputs
        st.markdown("**Answer:**")
        formatted_answer = format_text_with_actions(example["answer"])
        st.markdown(formatted_answer)

    elif swap_type == "output_swap":
        # Two different answers
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Chosen Answer:**")
            formatted_chosen = format_text_with_actions(example["chosen_answer"])
            st.markdown(formatted_chosen)

        with col2:
            st.markdown("**Rejected Answer:**")
            formatted_rejected = format_text_with_actions(example["rejected_answer"])
            st.markdown(formatted_rejected)


def show_evaluation_guide(example):
    """Show concise instructions for the current sample format."""
    swap_type = example.get("swap_type", "")
    if swap_type == "input_swap":
        format_note = (
            "This sample has **Chosen Video + Rejected Video** and a **single answer**. "
            "Judge correctness using the **single answer with the Chosen Video**."
        )
    else:
        format_note = (
            "This sample has a **single video** and **Chosen Answer + Rejected Answer**. "
            "Judge correctness using the **Chosen Answer with the video**."
        )

    st.info(
        "\n".join(
            [
                "### What to do",
                "1. **Part 1 - Visual check:** look at the visual input (video/image).",
                "2. **Part 2 - Answer check:** read the answer text.",
                "3. **Correctness:** evaluate if the target answer is correct for the chosen visual input.",
                "4. **Visual quality:** evaluate only the visual quality of the chosen visual input.",
                "",
                format_note,
            ]
        )
    )


def main():
    # Load data first to check if files exist
    original_data = load_jsonl(ORIGINAL_FILE)
    progress_data = load_progress()

    if not original_data:
        st.error(f"Could not load data from {ORIGINAL_FILE}")
        return

    # Check if user has entered alias
    user_alias = get_user_alias()
    if not user_alias:
        # Show alias input form
        entered_alias = show_alias_input()
        if entered_alias:
            set_user_alias(entered_alias, progress_data)
            st.rerun()
        return

    # Main app interface
    st.title("🔍 Data Evaluation App V2")
    st.markdown("---")

    # Get user session
    session_id = get_user_session_id()

    # Display progress in sidebar
    total_examples = len(original_data)
    evaluated_count = len(progress_data["evaluated_ids"])
    remaining_count = total_examples - evaluated_count
    
    # Get user stats
    user_stats = get_user_stats(user_alias, progress_data)
    active_users = get_active_users_count(progress_data)

    # Enhanced sidebar with dual progress
    st.sidebar.header("📊 Global Progress")
    st.sidebar.metric("Total Examples", total_examples)
    st.sidebar.metric("Evaluated", evaluated_count)
    st.sidebar.metric("Remaining", remaining_count)
    st.sidebar.progress(evaluated_count / total_examples if total_examples > 0 else 0)

    st.sidebar.markdown("---")
    st.sidebar.header(f"👤 Your Progress ({user_alias})")
    st.sidebar.metric("You've labeled", user_stats["total_labeled"])
    st.sidebar.metric("Your sessions", len(user_stats["sessions"]))
    
    if evaluated_count > 0:
        contribution_pct = (user_stats["total_labeled"] / evaluated_count) * 100
        st.sidebar.metric("Your contribution", f"{contribution_pct:.1f}%")

    st.sidebar.markdown("---")
    st.sidebar.metric("👥 Active Users", active_users)
    st.sidebar.write(f"**Session ID:** `{session_id[:8]}...`")

        # --- Admin: export results ---
    # --- Public export results ---
    st.sidebar.markdown("---")
    st.sidebar.header("📦 Export Results")

    # 1) Download labeled JSONL
    if os.path.exists(LABELED_FILE):
        with open(LABELED_FILE, "rb") as f:
            st.sidebar.download_button(
                "⬇️ Scarica labeled JSONL",
                data=f,
                file_name=Path(LABELED_FILE).name,
                mime="application/json",
            )
    else:
        st.sidebar.warning("LABELED_FILE non trovato sul server.")

    # 2) Download progress JSON
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "rb") as f:
            st.sidebar.download_button(
                "⬇️ Scarica progress JSON",
                data=f,
                file_name=Path(PROGRESS_FILE).name,
                mime="application/json",
            )
    else:
        st.sidebar.warning("PROGRESS_FILE non trovato sul server.")

    # Get next example
    current_example, current_index = get_next_example(original_data, progress_data, session_id)

    if current_example is None:
        st.success("🎉 You've completed your assignment of 30 samples!")
        st.balloons()
        
        # Show final stats
        st.subheader("📈 Your Session Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Samples Completed", 30)
            st.metric("Your Total Contribution", user_stats["total_labeled"])
        with col2:
            st.metric("Active Contributors", active_users)
            st.metric("Global Progress", f"{evaluated_count} / {total_examples}")
        
        st.markdown("---")
        
        # Option to start a new batch
        st.subheader("🔄 Start Another Batch?")
        st.write("Would you like to evaluate another 30 samples?")
        
        if st.button("✨ Start New Batch", type="primary", use_container_width=True):
            # Clear current assignment to generate a new one
            if session_id in progress_data["user_sessions"]:
                # Remove old assignment
                old_assignment = progress_data["user_sessions"][session_id].get("assignment", [])
                
                # Clear the assignment to trigger new generation
                progress_data["user_sessions"][session_id].pop("assignment", None)
                progress_data["user_sessions"][session_id]["position"] = 0
                progress_data["user_sessions"][session_id]["labeled_count"] = 0
                
                save_progress(progress_data)
                st.success("🎯 Generating new assignment...")
                st.rerun()
        
        return

    # Get task info for display
    task_family, task_variant, preference_type, video_id = parse_sample_id(current_example['id'])
    task_display = f"{task_family}_{task_variant}_{preference_type}"
    
    # Display progress bar at the top
    progress_percentage = (current_index + 1) / 30
    st.progress(progress_percentage, text=f"Progress: {current_index + 1} / 30 samples completed")
    
    # Display current example info with assignment progress
    st.write(f"**Sample {current_index + 1} of 30** ({task_display}) | ID: `{current_example['id']}`")
    if preference_type == "out":
        st.info(f"Experiment 1/2 (Output preference): single-video samples ({current_index + 1}/15)")
    else:
        st.info(f"Experiment 2/2 (Input preference): single-answer samples ({current_index - 14}/15)")
    show_evaluation_guide(current_example)

    st.subheader("Visual Input")
    display_media_content(current_example)

    st.markdown("---")

    st.subheader("Answer Text")
    display_text_content(current_example)

    st.markdown("---")

    # Evaluation scales
    st.subheader("🏷️ Evaluation")
    st.info("Submit two ratings: correctness of the target answer with the chosen visual input, and visual quality of the chosen visual input.")

    correctness_key = f"correctness_{current_example['id']}"
    quality_key = f"quality_{current_example['id']}"
    swap_type = current_example.get("swap_type", "")

    if swap_type == "output_swap":
        correctness_prompt = "-- Is the CHOSEN answer correct with respect to the video/image? --"
    else:
        correctness_prompt = "-- Is the answer correct with respect to the CHOSEN video/image? --"

    correctness_options = {
        correctness_prompt: None,
        "Wrong": "wrong",
        "Ambiguous": "ambiguous",
        "Correct": "correct"
    }
    quality_options = {
        "-- Select visual quality --": None,
        "Very Poor": "very_poor",
        "Poor": "poor",
        "Acceptable": "acceptable",
        "Good": "good",
        "Excellent": "excellent"
    }

    selected_correctness_display = st.radio(
        "Correctness",
        list(correctness_options.keys()),
        index=0,
        horizontal=True,
        key=correctness_key,
        help="Use the chosen visual input as reference. For output_swap, evaluate the chosen answer."
    )
    selected_quality_display = st.radio(
        "Visual Quality",
        list(quality_options.keys()),
        index=0,
        horizontal=True,
        key=quality_key,
        help="Evaluate the visual quality of the video/image."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Evaluation", type="primary", use_container_width=True):
            selected_correctness = correctness_options[selected_correctness_display]
            selected_quality = quality_options[selected_quality_display]

            if selected_correctness is None or selected_quality is None:
                st.error("Please select one option for both Correctness and Visual Quality.")
            else:
                evaluate_example(current_example, selected_correctness, selected_quality, progress_data)

    with col2:
        if st.button("⏭️ Skip", use_container_width=True):
            skip_example(current_example, progress_data)

    st.markdown("---")

    # Download button at the bottom (smaller)
    try:
        zip_data, zip_filename = prepare_sample_download(current_example)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.download_button(
                label="📥 Download Sample",
                data=zip_data,
                file_name=zip_filename,
                mime="application/zip",
                help="Download all media files, prompt, and answers for this sample"
            )
    except Exception as e:
        st.error(f"Error preparing download: {str(e)}")


def derive_evaluation_label(correctness_label, visual_quality_label):
    """Derive legacy evaluation_label from split correctness and quality labels."""
    if correctness_label == "correct" and visual_quality_label in {"acceptable", "good", "excellent"}:
        return "good_labeled"
    if correctness_label == "wrong":
        return "wrong_labeled"
    if correctness_label == "ambiguous":
        return "ambiguous"
    if visual_quality_label in {"very_poor", "poor"} and correctness_label != "wrong":
        return "bad_visual_quality"
    return "ambiguous"


def evaluate_example(example, correctness_label, visual_quality_label, progress_data):
    """Save evaluation and move to next example"""
    session_id = st.session_state.user_session_id
    user_alias = get_user_alias()
    evaluation_label = derive_evaluation_label(correctness_label, visual_quality_label)
    
    # Add evaluation label and timestamp
    labeled_example = example.copy()
    labeled_example["correctness_label"] = correctness_label
    labeled_example["visual_quality_label"] = visual_quality_label
    labeled_example["evaluation_label"] = evaluation_label
    labeled_example["evaluation_timestamp"] = datetime.now().isoformat()
    labeled_example["evaluator_session"] = session_id
    labeled_example["evaluator_alias"] = user_alias

    # Save to labeled file
    save_to_jsonl(labeled_example, LABELED_FILE)

    # Update progress
    progress_data["evaluated_ids"].append(example["id"])
    
    # Advance position in assignment
    if session_id in progress_data["user_sessions"]:
        current_position = progress_data["user_sessions"][session_id].get("position", 0)
        progress_data["user_sessions"][session_id]["position"] = current_position + 1
        progress_data["user_sessions"][session_id]["labeled_count"] = progress_data["user_sessions"][session_id].get("labeled_count", 0) + 1
    
    # Update alias stats
    if user_alias in progress_data["alias_stats"]:
        progress_data["alias_stats"][user_alias]["total_labeled"] = progress_data["alias_stats"][user_alias].get("total_labeled", 0) + 1
    
    save_progress(progress_data)

    # Show success message and rerun
    st.success(
        f"✅ Saved | Correctness: **{correctness_label.replace('_', ' ').title()}** | "
        f"Quality: **{visual_quality_label.replace('_', ' ').title()}** | "
        f"Legacy label: **{evaluation_label.replace('_', ' ').title()}**"
    )
    st.rerun()


def skip_example(example, progress_data):
    """Skip current example and move to next"""
    session_id = st.session_state.user_session_id
    
    # Advance position in assignment without labeling
    if session_id in progress_data["user_sessions"]:
        current_position = progress_data["user_sessions"][session_id].get("position", 0)
        progress_data["user_sessions"][session_id]["position"] = current_position + 1
    
    save_progress(progress_data)

    st.info("⏭️ Example skipped")
    st.rerun()


if __name__ == "__main__":
    main()
