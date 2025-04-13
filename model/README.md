## Sources

- <https://parkerdixon.github.io//Squash-Vision/> expensive CNN approach
- <https://repository.tudelft.nl/file/File_ce8142e4-e681-43f0-80f5-f9ec6c2ae975?preview=1> very good
- <https://github.com/veedlaw/squash-drive-analyst> just python code
- <https://web.archive.org/web/20190301054438/http://felday.info/projects/2016/08/16/Squash-Ball-Tracking.html>  Slow
  code and in matlab
- <https://www.cis.upenn.edu/wp-content/uploads/2019/08/HonorspaperforEAS499_Wu_Judd.pdf>  No code but a lot of
  description, also matlab
- <https://mikecann.blog/posts/squash-lasers-and-other-thoughts> interesting overview of approaches

## Tips

### Preprocessing

- use a simpler foreground detector, not mixed gaussians, just a simple threshold
- erosion followed by dilation to remove noise (radius 1, param 4)
- boundary filling

### Player detection

- for players, need at least 300 square pixels
- use center of bounding box, not blob centroid for position.
- the centers that are farthest apart are the players
- assign all blobs to a list of either player 1 or player 2 depending on the distance to the two large player blobs
- Merge them all and use the average of the merged blobs as the player position

### Ball Detection

- morphological operations do not work here
- Even with the blur, it is likely still the most round object.
- color does neither as it often blurs into almost transparent

### Tracking

Data Structure:

- id: the integer id of the track
- bbox: the bounding box of the track
- kalmanFilter: the object used for motion prediction
- age: # of frames since the track was first detected
- totalVisibleCount: # of total frames in which the track was detected
- consecutiveInvisibleCount: # of consecutive frames where the track was not detected

#### Kalman Filter

use constant velocity model with a high motion noice (100, 25) to rely on the measurements more than the prediction

#### Maintenance

A track is created for each relevant blob.
We assign the detected ball blob to the track whose predicted position is closest to the detected position.

## TODO

### Ball Detection

- Count in how many frames the ball is detected. If it is not detected for 10 frames, reinitialize the tracking object.
  Also the percentage of detection should be >75%