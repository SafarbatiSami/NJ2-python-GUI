import tkinter as tk
from tkinter import filedialog , IntVar , N, W, E , S
from compute_params import compute_directory


class SimpleGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Compute Parameters GUI")
        self.master.geometry("600x235")  # Set the window size

        # Folder paths
        self.nuclei_folder_path = tk.StringVar()
        self.chromocenter_folder_path = tk.StringVar()

        # Widgets
        self.create_widgets()

    def create_widgets(self):
        
        self.master.columnconfigure(0, weight=1)
        self.master.columnconfigure(1, weight=1)


        global compute_chromocenters_state
        
        # Nuclei Folder Entry
        nuclei_folder_entry = tk.Entry(self.master, textvariable=self.nuclei_folder_path, width=30)
        nuclei_folder_entry.grid(row=1, column=0, columnspan=2, pady=10, sticky = N )

        # Nuclei Folder Button
        nuclei_folder_button = tk.Button(
            self.master, text="Open Nuclei Folder", command=self.get_nuclei_folder
        )
        nuclei_folder_button.grid(row=2, column=0, columnspan=2, pady=5 ,sticky = N)
        
        # Chromocenter Folder Entry
        chromocenter_folder_entry = tk.Entry(self.master, textvariable=self.chromocenter_folder_path, width=30)
        
        # Chromocenter Folder Button
        chromocenter_folder_button = tk.Button(
            self.master, text="Open Chromocenter Folder", command=self.get_chromocenter_folder
        )
       
        # Run Button
        run_button = tk.Button(self.master, text="Run", command=self.run_process)
        run_button.grid(row=5, column=0, columnspan=2, pady=10, sticky = S)
        
        def display_chromo_filedialog():
            if compute_chromocenters_state.get() :
                chromocenter_folder_entry.grid(row=3, column=0, columnspan=2, pady=10, sticky = N )
                chromocenter_folder_button.grid(row=4, column=0, columnspan=2, pady=5, sticky = N)
            else : 
                chromocenter_folder_entry.grid_remove()
                chromocenter_folder_button.grid_remove()
                
        compute_chromocenters_state = IntVar(value=0)
        chromo_check_button = tk.Checkbutton(self.master, text=" Compute Chromocenters parameters  ? ", command= display_chromo_filedialog ,variable=compute_chromocenters_state)
        chromo_check_button.grid(row=0, column=0, columnspan=2, pady=10)
        
   
                
                
            
            
        
    def get_nuclei_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.nuclei_folder_path.set(folder_path)
        return folder_path

    def get_chromocenter_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.chromocenter_folder_path.set(folder_path)
        
        return folder_path
    def run_process(self):
        nuclei_path = self.nuclei_folder_path.get()
        if compute_chromocenters_state.get():
            chromocenter_path = self.chromocenter_folder_path.get()
        else :
            chromocenter_path = None
        print(f"Running process with Nuclei Folder: {nuclei_path}, Chromocenter Folder: {chromocenter_path}")
        
        ss = [0.1032, 0.1032, 0.2] 
        compute_directory(path=nuclei_path,
            cc_path=chromocenter_path,
            spacing=ss)
       
if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleGUI(root)
    root.mainloop()
