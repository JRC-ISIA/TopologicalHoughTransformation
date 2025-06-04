import cv2


class ImageSmoothing:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def smooth_image(self, image):
        # Lade das Bild
       # image = cv2.imread(image_path)
        self.image = image
        # Überprüfe, ob das Bild geladen wurde
        if image is None:
            print("Fehler: Bild konnte nicht geladen werden!")
            return None

        # Wende den Glättungskern auf das Bild an
        smoothed_image = cv2.blur(image, (self.kernel_size, self.kernel_size))

        return smoothed_image
