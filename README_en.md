# *C. elegans* Simulator  

This [*Caenorhabditis elegans* (*C. elegans*) simulator](https://wormsim.streamlit.app/) was developed to reproduce salt chemotaxis behavior based on salt concentration memory. The model is built upon the following research paper:  

**Reference Paper:**  
Hironaka Masakatsu, Sumi Tomonari “A neural network model that generates salt concentration memory-dependent chemotaxis in *Caenorhabditis elegans*” *eLife* **14**:RP104456 (2025).
[https://elifesciences.org/articles/104456](https://elifesciences.org/articles/104456)

This simulator is available as a web application, allowing users to intuitively configure input parameters and observe *C. elegans* movements through animations.  

## **How to Use**  

1. Open the [simulator](https://wormsim.streamlit.app/) and select *C. elegans* individuals and salt concentration functions from the tabs below.  
2. Change the parameters.  
3. Click **Run Simulation** to display the animation on the screen.  

## **Technical Features**  

- A **Rust-based** [simulation engine](https://github.com/118-Oganesson/wormsim_rs) significantly reduces computation time.  
- **Plotly** generates interactive animations.  
- **Streamlit** provides a simple and user-friendly interface for input and result manipulation.  

This simulator helps visualize the behavioral mechanisms of *C. elegans*, making it useful for academic research, education, and even playful experiments. Give it a try!
