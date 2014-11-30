var margin = {top: 20, right: 20, bottom: 30, left: 40},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;

/* 
 * value accessor - returns the value to encode for a given data object.
 * scale - maps value to a visual display encoding, such as a pixel position.
 * map function - maps from data value to display value
 * axis - sets up axis
 */ 

// setup x 
var xValue = function(d) { return d.XAxis;}, // data -> value
    xScale = d3.scale.linear().range([0, width]), // value -> display
    xMap = function(d) { return xScale(xValue(d));}, // data -> display
    xAxis = d3.svg.axis().scale(xScale).orient("bottom");

// setup y
var yValue = function(d) { return d["YAxis"];}, // data -> value
    yScale = d3.scale.linear().range([height, 0]), // value -> display
    yMap = function(d) { return yScale(yValue(d));}, // data -> display
    yAxis = d3.svg.axis().scale(yScale).orient("left");

// setup fill color
var cValue = function(d) { return d.Category;}, 
color = d3.scale.category20().domain(d3.range(10));

var radius = 3.5;

// add the graph canvas to the body of the webpage
var svg = d3.select("body").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// add the tooltip area to the webpage
var tooltip = d3.select("body").append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);

// load data
d3.csv("data/temp_data.csv", function(error, data) {

  // change string (from CSV) into number format
  data.forEach(function(d) {
    d.XAxis = +d.XAxis;
    d["YAxis"] = +d["YAxis"];
//    console.log(d);
  });

  // don't want dots overlapping axis, so add in buffer to data domain
  xScale.domain([d3.min(data, xValue)-1, d3.max(data, xValue)+1]);
  yScale.domain([d3.min(data, yValue)-1, d3.max(data, yValue)+1]);

  // x-axis
  svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis)
    .append("text")
      .attr("class", "label")
      .attr("x", width)
      .attr("y", -6)
      .style("text-anchor", "end")
      .text("Axis 1");

  // y-axis
  svg.append("g")
      .attr("class", "y axis")
      .call(yAxis)
    .append("text")
      .attr("class", "label")
      .attr("transform", "rotate(-90)")
      .attr("y", 6)
      .attr("dy", ".71em")
      .style("text-anchor", "end")
      .text("Axis 2");

  // draw dots
  svg.selectAll(".dot")
      .data(data)
    .enter()
    .append('g').attr("class", "state")
    .append("circle")
      .attr("class", "dot")
      .attr("r", 3.5)
      .attr("cx", xMap)
      .attr("cy", yMap)
      .style("fill", function(d) { return color(cValue(d));})
      .on("mouseover", function(d) {
          tooltip.transition()
               .duration(200)
               .style("opacity", 0.95);
          tooltip.html(d["Name"] + "<br/> (" + xValue(d) 
	        + ", " + yValue(d) + ")")
               .style("left", (d3.event.pageX + 5) + "px")
               .style("background-color", "#F0F0F0")
               .style("top", (d3.event.pageY - 28) + "px");
      })
      .on("mouseout", function(d) {
          tooltip.transition()
               .duration(500)
               .style("opacity", 0);    
      });

  // draw legend
  var legend = svg.selectAll(".legend")
      .data(color.domain())
    .enter().append("g")
      .attr("class", "legend")
      .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });

  // draw legend colored rectangles
  legend.append("rect")
      .attr("x", width - 18)
      .attr("width", 18)
      .attr("height", 18)
      .style("fill", color);

  // draw legend text
  legend.append("text")
      .attr("x", width - 24)
      .attr("y", 9)
      .attr("dy", ".35em")
      .style("text-anchor", "end")
      .text(function(d) { return d;})
});

window.svg = d3.select("svg")

/*
TUTORIAL - 
0 = No rectangle
1 = Drawing rectangle
2 = Finished drawing
*/
var isOn = 0;

svg
.on( "mousedown", function() {
    console.log("isOn",isOn);
    var p = d3.mouse(this);
    var mouse_x = p[0];
    var mouse_y = p[1];

    if (isOn==2){
	
	var s = svg.select( "rect.selection");
	var d = {
                x       : parseInt( s.attr( "x"), 10),
                y       : parseInt( s.attr( "y"), 10),
                width   : parseInt( s.attr( "width"), 10),
                height  : parseInt( s.attr( "height"), 10)
	};

	var mouseInRectangle = ((mouse_x > d.x) && (mouse_x < (d.x + d.width)) && 
				(mouse_y > d.y) && (mouse_y < (d.y + d.height)));

	if (mouseInRectangle) {
	    var tmp = svg.selectAll("g.state.selected");
	    console.log(JSON.stringify(tmp, null, 4));
	    console.log("Sending request!"); 
	    $.ajax("http://0.0.0.0:8080/post", {
		data : JSON.stringify(tmp),
		type : "POST"
	    });
	}
	svg.selectAll( "rect.selection").remove();
	d3.selectAll( 'g.state.selection').classed( "selection", false);
	d3.selectAll( 'g.selected').classed("selected", false);   

    } else if (isOn==0){

	svg.selectAll( "rect.selection").remove();
	d3.selectAll( 'g.state.selection').classed( "selection", false);
	var p = d3.mouse( this);
	svg.append("rect")
	    .attr({
		rx      : 6,
		ry      : 6,
		class   : "selection",
		x       : p[0],
		y       : p[1],
		width   : 0,
		height  : 0
	    });

    } else if(isOn==1) {
	var s = svg.select( "rect.selection");
	var d = {
                x       : parseInt( s.attr( "x"), 10),
                y       : parseInt( s.attr( "y"), 10),
                width   : parseInt( s.attr( "width"), 10),
                height  : parseInt( s.attr( "height"), 10)
	};
	    d3.selectAll('g.state.selection.selected').classed( "selected", false);
	    console.log("selecting");
	    console.log(JSON.stringify(tmp, null, 4));
	    d3.selectAll('.dot').each( 
		function( state_data, i) {
		    var cx = parseInt(d3.select(this).attr("cx"));
		    var cy = parseInt(d3.select(this).attr("cy"));
		    // check to see if circle inside selection frame
		    if(!d3.select( this).classed( "selected") && 
                       cx-radius>=d.x && cx+radius<=d.x+d.width && 
                       cy-radius>=d.y && cy+radius<=d.y+d.height) 
		    {		
			d3.select( this.parentNode)
			    .classed( "selection", true)
			    .classed( "selected", true);
			console.log("adding selection");
		    }
		}
	    );
    }
    isOn = (isOn+1)%3;
})
.on( "mousemove", function() {
    var s = svg.select( "rect.selection");
    if( !s.empty() && isOn==1) {
        var p = d3.mouse(this),
            d = {
                x       : parseInt( s.attr( "x"), 10),
                y       : parseInt( s.attr( "y"), 10),
                width   : parseInt( s.attr( "width"), 10),
                height  : parseInt( s.attr( "height"), 10)
            },
            move = {
                x : p[0] - d.x,
                y : p[1] - d.y
            }
        ;
        if( move.x < 1 || (move.x*2<d.width)) {
            d.x = p[0];
            d.width -= move.x;
        } else {
            d.width = move.x;       
        }

        if( move.y < 1 || (move.y*2<d.height)) {
            d.y = p[1];
            d.height -= move.y;
        } else {
            d.height = move.y;       
        }
        s.attr( d);
    }
})
.on( "mouseup", function() {
       // remove selection frame
});
