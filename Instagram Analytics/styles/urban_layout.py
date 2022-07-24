# Styles guide from https://urbaninstitute.github.io/graphics-styleguide/

# Color palettes
## Single colors
CYAN    = '#1696d2'
GRAY    = '#d2d2d2'
BLACK   = '#000000'
YELLOW  = '#fdbf11'
MAGENTA = '#ec008b'
GREEN   = '#55b748'
RED     = '#db2b27'
OCEAN   = '#0a4c6a'
DARK    = '#332d2f'

## Color shades
CYAN_SHADES    = ["#CFE8F3","#A2D4EC","#73BFE2","#46ABDB","#1696D2","#12719E","#0A4C6A","#062635"]
GRAY_SHADES    = ["#D5D5D4","#ADABAC","#848081","#5C5859","#332D2F","#262223","#1A1717","#0E0C0D"]
YELLOW_SHADES  = ["#FFF2CF","#FCE39E","#FDD870","#FCCB41","#FDBF11","#E88E2D","#CA5800","#843215"]
MAGENTA_SHADES = ["#F5CBDF","#EB99C2","#E46AA7","#E54096","#EC008B","#AF1F6B","#761548","#351123"]
GREEN_SHADES   = ["#DCEDD9","#BCDEB4","#98CF90","#78C26D","#55B748","#408941","#2C5C2D","#1A2E19"]
RED_SHADES     = ["#F8D5D4","#F1AAA9","#E9807D","#E25552","#DB2B27","#A4201D","#6E1614","#370B0A"]

## Categorical groups
CATEGORICAL_GROUPS = {
    2 : [CYAN, YELLOW],
    3 : [CYAN, YELLOW, BLACK],
    4 : [BLACK, GRAY, YELLOW, CYAN],
    5 : [CYAN, OCEAN, GRAY, YELLOW, DARK],
    6 : [CYAN, GRAY, MAGENTA, YELLOW, DARK, OCEAN],
    'political': [CYAN, RED]
}

## Heatmap continuous scale
HEATMAP_SCALE = ['#ca5800', '#fdbf11', '#fdd870', '#fff2cf', '#cfe8f3', '#73bfe2', '#1696d2', '#0a4c6a']


# Plotly layout specs
LAYOUT_SPECS = {
    'title' : {
        'font_family' : 'Lato-Bold',
        'font_color' : '#000000',
        'font_size' : 18
    },
    'hoverlabel' : {
        'font_family' : 'Lato-Regular',
        'font_size' : 14,
    },
    'legend' : {
        'title_font_family' : 'Lato-Bold',
        'title_font_color' : '#000000',
        'title_font_size' : 14,
        'font_family' : 'Lato-Regular',
        'font_color' : '#000000',
        'font_size' : 13,
    },
    'xaxis' : {
        'titlefont_family' : 'Lato-Italic',
        'titlefont_color' : '#000000',
        'titlefont_size' : 14,
        'tickfont_family' : 'Lato-Regular',
        'tickfont_color' : '#000000',
        'tickfont_size' : 12,
    },
    'yaxis' : {
        'titlefont_family' : 'Lato-Italic',
        'titlefont_color' : '#000000',
        'titlefont_size' : 14,
        'tickfont_family' : 'Lato-Regular',
        'tickfont_color' : '#000000',
        'tickfont_size' : 12,
    },
}