import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageTk

class ReCAMPyTorchModel(nn.Module):
    def __init__(self, num_classes):
        super(ReCAMPyTorchModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ReCAMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Re-CAM Mini-Project")

        # Initialize variables
        self.target_class_index = tk.StringVar()
        self.target_class_index.set("")
        self.target_class_index_value = None

        # Create GUI elements
        self.build_gui()

        # Initialize ReCAM model
        self.model = None

    def build_gui(self):
        # Configure root window for full screen
        # self.root.attributes("-fullscreen", True)

        input_frame = tk.Frame(self.root, bg="#336699")  # Set background color for topmost bar
        input_frame.pack(side="top", fill="x")

        # Input elements
        tk.Label(input_frame, text="Target Class Index:", bg="#336699", fg="white").pack(side="left", padx=(10, 5), pady=5)
        self.target_class_entry = tk.Entry(input_frame, textvariable=self.target_class_index)
        self.target_class_entry.pack(side="left", padx=(0, 5), pady=5)
        submit_button = tk.Button(input_frame, text="Submit", command=self.submit_target_class)
        submit_button.pack(side="left", padx=(0, 10), pady=5)

        # Train Model button
        train_button = tk.Button(input_frame, text="Train Model", command=self.train_model)
        train_button.pack(side="left", padx=(0, 10), pady=5)

        # Fine-tune Model button
        fine_tune_button = tk.Button(input_frame, text="Fine-tune Model", command=self.fine_tune_model)
        fine_tune_button.pack(side="left", padx=(0, 10), pady=5)

        # Select Dataset Directory button
        select_dir_button = tk.Button(input_frame, text="Select Dataset Directory", command=self.load_and_preprocess_data)
        select_dir_button.pack(side="left", padx=(0, 10), pady=5)

        # Generate ReCAM button
        generate_button = tk.Button(input_frame, text="Generate ReCAM", command=self.generate_ReCAM)
        generate_button.pack(side="left", padx=(0, 10), pady=5)

        # Save ReCAM button
        save_button = tk.Button(input_frame, text="Save ReCAM", command=self.save_ReCAM)
        save_button.pack(side="left", pady=5)

        # Frame for window controls
        # control_frame = tk.Frame(self.root, bg="lightgray")
        # control_frame.pack(side="top", fill="x", pady=5)
        #
        # # Minimize button
        # min_button = tk.Button(control_frame, text="_", width=3, command=self.minimize_window)
        # min_button.pack(side="right", padx=(0, 10))
        #
        # # Maximize or Restore Down button
        # self.max_restore_icon = tk.PhotoImage(file="maximize.png")
        # self.restore_icon = tk.PhotoImage(file="restore.png")
        # self.max_restore_button = tk.Button(control_frame, image=self.max_restore_icon, width=30, height=20, command=self.maximize_restore_window)
        # self.max_restore_button.pack(side="right")
        #
        # # Exit button
        # exit_button = tk.Button(control_frame, text="X", width=3, command=self.exit_application)
        # exit_button.pack(side="right", padx=(0, 10))

        # CAM Display
        self.cam_display_frame = tk.Frame(self.root, bg="blue")  # Set background color for the CAM Display frame
        self.cam_display_frame.pack(expand=True, fill="both")

        self.cam_display = tk.Label(self.cam_display_frame, bg="white")
        self.cam_display.pack(expand=True, fill="both")

    # def minimize_window(self):
    #     self.root.iconify()
    #
    # def maximize_restore_window(self):
    #     if self.root.attributes("-fullscreen"):
    #         self.root.attributes("-fullscreen", False)
    #         self.max_restore_button.config(image=self.max_restore_icon)
    #     else:
    #         self.root.attributes("-fullscreen", True)
    #         self.max_restore_button.config(image=self.restore_icon)
    #
    # def exit_application(self):
    #     self.root.destroy()


    def submit_target_class(self):
        value = self.target_class_index.get()
        if value.isdigit():
            self.target_class_index.set("")  # Clear entry after submission
            self.target_class_index_value = int(value)
            messagebox.showinfo("Success", f"Target class index set to {self.target_class_index_value}")
        else:
            messagebox.showerror("Error", "Please enter a valid integer value for the target class index.")



    def load_and_preprocess_data(self):
        # Prompt user to select dataset directory
        dataset_dir = filedialog.askdirectory(initialdir="/", title="Select Dataset Directory")

        if dataset_dir:
            # Define image data generators with preprocessing
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1.0/255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True
            )
            val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

            # Load and preprocess training data
            train_data = train_datagen.flow_from_directory(
                os.path.join(dataset_dir, 'train'),
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical'
            )

            # Load and preprocess validation data
            val_data = val_datagen.flow_from_directory(
                os.path.join(dataset_dir, 'val'),
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical'
            )

            return train_data, val_data
        else:
            return None, None



    def build_model(self, num_classes):
        # Your model building code here
        self.model = ReCAMPyTorchModel(num_classes)
        # Load pre-trained weights if needed
        # self.model.load_state_dict(torch.load("pretrained_weights.pth"))

    def train_model(self):
        # Load and preprocess the dataset
        train_data, val_data = self.load_and_preprocess_data()

        if train_data and val_data:
            # Define the number of classes in your dataset
            num_classes = len(train_data.class_indices)

            # Build the ReCAM model
            self.model = self.build_model(num_classes)

            # Compile the model
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # Train the model
            self.model.fit(train_data, epochs=10, validation_data=val_data)
            print("Model training completed.")
        else:
            print("Dataset directory not selected.")

    def fine_tune_model(self):
        # Load and preprocess the dataset
        train_data, val_data = self.load_and_preprocess_data()

        if train_data and val_data:
            # Fine-tune the entire model
            self.model.trainable = True
            self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5),
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])

            # Train the model
            self.model.fit(train_data, epochs=5, validation_data=val_data)
            print("Model fine-tuning completed.")
        else:
            print("Dataset directory not selected.")
    def generate_ReCAM(self):
        # Load the image
        image_path = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=[("Image Files", "*.jpg *.png")])
        if image_path:
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0)

            # Get the target class index from the entry field
            target_class_index = int(self.target_class_entry.get())

            # Generate ReCAM
            if self.model is not None:
                cam_img = self.generate_ReCAM_image(self.model, img, target_class_index)
                if cam_img is not None:
                    # Display ReCAM image
                    self.display_ReCAM(cam_img)
                else:
                    print("The top predicted class is not the target class.")
            else:
                print("Model is not trained yet.")

    def generate_ReCAM_image(self, model, img, target_class_index):
        # Convert numpy array to PyTorch tensor
        img_tensor = torch.from_numpy(img).permute(0, 3, 1, 2).float()

        # Forward pass
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            output = model(img_tensor)

        # Get the predicted class scores
        pred_scores = torch.softmax(output, dim=1)

        # Get the predicted class index with the highest score
        _, predicted_class_index = torch.max(pred_scores, 1)

        # Check if the predicted class matches the target class index
        if predicted_class_index.item() != target_class_index:
            return None  # Return None if the predicted class is not the target class

        # Get the feature maps from the final convolutional layer
        feature_maps = model.features(img_tensor)

        # Compute the class activation map (CAM)
        cam = torch.matmul(model.fc.weight[target_class_index], feature_maps.squeeze())
        cam = torch.nn.functional.relu(cam)  # Apply ReLU to remove negative values
        cam = cam.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        # Normalize the CAM
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        # Resize CAM to match the image size
        cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)

        # Convert CAM to numpy array
        cam = cam.squeeze().cpu().numpy()

        # Apply colormap
        cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        # Convert image to RGB
        img_rgb = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)

        # Overlay CAM on the original image
        cam_img = cv2.addWeighted(img_rgb, 0.7, cam, 0.3, 0)

        return cam_img

    def display_ReCAM(self, cam_img):
        # Convert BGR to RGB (OpenCV uses BGR by default)
        cam_img_rgb = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)

        # Convert to ImageTk format
        cam_img_tk = Image.fromarray(cam_img_rgb)

        # Update the label to display ReCAM image
        self.cam_display.img = ImageTk.PhotoImage(cam_img_tk)
        self.cam_display.config(image=self.cam_display.img)


    def save_ReCAM(self):
        # Ensure ReCAM image is generated before saving
        if hasattr(self, 'cam_display') and hasattr(self.cam_display, 'img'):
            # Convert PIL Image to OpenCV format (BGR)
            cam_img_bgr = cv2.cvtColor(np.array(self.cam_display.img), cv2.COLOR_RGB2BGR)

            # Ask user for saving location
            save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg")])
            if save_path:
                # Save the ReCAM image
                cv2.imwrite(save_path, cam_img_bgr)
                print("ReCAM image saved successfully.")
        else:
            print("No ReCAM image to save.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ReCAMApp(root)
    # root.attributes('-fullscreen', True)
    # # root.attributes('-zoomed', True)  # For Windows systems
    # root.attributes('-topmost', True)  # Make the window appear on top
    # # Configure the window properties to keep the standard controls
    # root.wm_attributes('-fullscreen', True)
    # root.wm_attributes('-topmost', True)

    root.mainloop()
