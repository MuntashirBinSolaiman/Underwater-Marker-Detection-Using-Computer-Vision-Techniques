import os


class FrameLoader:
    def __init__(self, video_directory):
        self.video_directory = video_directory

    def get_image_files(self):
        all_files = os.listdir(self.video_directory)
        image_files = self._filter_image_files(all_files)
        return self._sort_files_numerically(image_files)

    def _filter_image_files(self, file_list):
        valid_extensions = [".jpg", ".jpeg"]
        return [f for f in file_list if os.path.splitext(f)[-1].lower() in valid_extensions]

    def _sort_files_numerically(self, file_list):
        return sorted(file_list, key=self._extract_index)

    def _extract_index(self, filename):
        name_part = os.path.splitext(filename)[0]
        return int(name_part)  # Assumes frame filenames are numeric
