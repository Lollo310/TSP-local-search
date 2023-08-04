import os

def process_txt_file(dir):
    for filename in sorted(os.listdir(dir)):
        if filename.endswith(".tsp"):
            filepath = os.path.join(dir, filename)
            
            # Read the content of the text file into a list of lines
            with open(filepath, "r") as file:
                lines = file.readlines()

            # Process the first 6 lines by adding '#'
            for i in range(min(6, len(lines))):
                line = lines[i]
                lines[i] = "# " + line  # Add '#' to the beginning of the line
                    
            # Replace double spaces with a single space throughout the entire text
            for i in range(len(lines)):
                lines[i] = " ".join(lines[i].split())
                
            # Join the lines back into a single string
            modified_text = "\n".join(lines)

            # Write the modified lines back to the text file
            with open(filepath, "w") as file:
                file.writelines(modified_text)


if __name__ == "__main__":
    input_directory = "./TSP_datasets/"  # Sostituisci con il percorso della tua directory
    process_txt_file(input_directory)