# Field Images Directory

This directory is for real-world field condition images, as opposed to controlled dataset images.

## Field Images vs. Controlled Images

### Controlled Images (PlantVillage)
- Clean backgrounds
- Consistent lighting
- Single leaf focus
- High quality
- Uniform conditions

### Field Images
- Natural backgrounds
- Variable lighting
- Multiple plants/leaves
- Variable quality
- Real-world conditions

## How to Capture Field Images

### Best Practices

1. **Lighting:**
   - Natural daylight is best
   - Avoid harsh shadows
   - Overcast days provide even lighting

2. **Distance:**
   - Fill the frame with the affected leaf/plant
   - Leave some margin around edges
   - Capture enough context

3. **Focus:**
   - Keep disease symptoms in focus
   - Avoid motion blur
   - Use camera's macro mode if available

4. **Angle:**
   - Shoot perpendicular to leaf surface
   - Avoid extreme angles
   - Capture the most affected area

5. **Background:**
   - Try to minimize background clutter
   - Focus on the diseased part
   - Natural backgrounds are acceptable

### Equipment Recommendations

- **Smartphone:** Modern smartphones (5MP+) work well
- **Digital Camera:** Any point-and-shoot or DSLR
- **Drone (Optional):** For large-scale field monitoring
- **Tripod (Optional):** For stability in low light

## Image Requirements

- **Format:** JPG or PNG
- **Resolution:** Minimum 224×224px (higher is better)
- **File Size:** Maximum 16MB
- **Color:** RGB (color images)

## Use Cases

### 1. Model Validation
Test how well models trained on controlled images perform on real field conditions

### 2. Domain Adaptation Research
Study the domain gap between lab and field images

### 3. Practical Deployment Testing
Validate the system with real-world agricultural use cases

### 4. Demonstration
Show the system working with realistic field images during thesis defense

## Expected Challenges

Field images may have:
- Lower prediction confidence
- More variation in results
- Need for domain adaptation techniques
- Background noise affecting visualizations

This is normal and demonstrates real-world application challenges!

## Example Field Image Scenarios

1. **Early Morning Inspection**
   - Dew on leaves
   - Soft morning light
   - Field_morning_tomato_01.jpg

2. **Midday Field Check**
   - Bright sunlight
   - Potential shadows
   - Field_midday_corn_01.jpg

3. **Disease Outbreak**
   - Multiple affected leaves
   - Varying disease stages
   - Field_outbreak_potato_01.jpg

4. **Drone Imagery**
   - Aerial view
   - Large field coverage
   - Drone_field_overview_01.jpg

## Testing Field Images

When analyzing field images:

1. Select "Field Images" in the dataset type dropdown
2. Use EfficientNet or Hybrid model for better generalization
3. Compare results with controlled images
4. Note confidence differences
5. Document challenges encountered

## Data Collection Tips

- Collect images at different times of day
- Capture various disease stages
- Include healthy plants for comparison
- Document metadata (location, date, weather)
- Maintain privacy (avoid identifying information)
