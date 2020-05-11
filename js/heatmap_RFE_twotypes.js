

var myFEATURES = [];
var myTYPES = [];

// set the dimensions and margins of the graph
var margin = {top: 30, right: 30, bottom: 30, left: 30},
  width = 1000 - margin.left - margin.right,
  height = 450 - margin.top - margin.bottom;

// GITHUBFIX
var datapath = "../";
try {
  d3.csv(datapath + "results_features_good/RFE_two_subtypes_freq.csv", function(data) {
    console.log(data)
  })
}
catch(err) {
  datapath = "../../"
  d3.csv(datapath + "results_features_good/RFE_two_subtypes_freq.csv", function(data) {
    console.log(data)})
}
d3.csv(datapath+"results_features_good/RFE_two_subtypes_freq.csv", function(data) {
var feature_array = [];
var type_array = [];


data.forEach(function(entry) {
    if (!feature_array.includes(entry.features)){
        feature_array.push(entry.features)
    };

    if (!type_array.includes(entry.type)){
    type_array.push(entry.type)
  };
});
var svg = d3.select("#my_dataviz")
.append("svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
.append("g")
  .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")");

// Labels of row and columns
var myGroups = feature_array
var myVars = type_array

// Build X scales and axis:
var x = d3.scaleBand()
  .range([ 0, width ])
  .domain(myGroups)
  .padding(0.01);
svg.append("g")
  .attr("transform", "translate(0," + height + ")")
  .call(d3.axisBottom(x))
  .selectAll("text")
      .style("text-anchor", "end")
      .style("font-size", "5px")
      .attr("dx", "-.8em")
      .attr("dy", ".15em")
      .attr("transform", "rotate(-65)");

// Build X scales and axis:
var y = d3.scaleBand()
  .range([ height, 0 ])
  .domain(myVars)
  .padding(0.01);
svg.append("g")
  .call(d3.axisLeft(y));


// Build color scale
var myColor = d3.scaleLinear()
  .range(["red", "green"])
  .domain([0,1]);
  var tooltip = d3.select("#my_dataviz")
      .append("div")
      .attr("class", "tooltip")
      .style("background-color", "white")
      .style("border", "solid")
      .style("border-width", "2px")
      .style("border-radius", "5px")
      .style("padding", "5px")
    // Three function that change the tooltip when user hover / move / leave a cell
    var mouseover = function(d) {
      tooltip
      d3.select(this)
        .style("stroke", "black")
    }
    var mousemove = function(d) {
      tooltip
        .html("Feature: " + d.features + " </b>" + " Frequency: " + d.freqs + "</b>" + " Accuracy: " + d.accuracy )
        .style("left", (d3.mouse(this[0] + 70) + "px"))
        .style("top", (d3.mouse(this)[1]) + "px")
    }
    var mouseleave = function(d) {
      tooltip
      d3.select(this)
        .style("stroke", "none")
    }

  svg.selectAll()
      .data(data)
      .enter()
      .append("rect")
      .attr("x", function(d) { return x(d.features) })
      .attr("y", function(d) { return y(d.type) })
      .attr("width", x.bandwidth() )
      .attr("height", y.bandwidth() )
      .style("fill", function(d) { return myColor(+d.accuracy)} )
          .on("mouseover", mouseover)
          .on("mousemove", mousemove)
          .on("mouseleave", mouseleave);



})

//
//
var myFEATURES = [];
var myTYPES = [];

// set the dimensions and margins of the graph
var margin = {top: 30, right: 30, bottom: 30, left: 30},
  width = 1000 - margin.left - margin.right,
  height = 450 - margin.top - margin.bottom;

// append the svg object to the body of the page

//Read the data
// d3.csv("https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/heatmap_data.csv", function(data) {
// console.log(data)})
var datapath = "../";
try {
  d3.csv(datapath +"results_features_good/RFE_two_subtypes_freq.csv",function(data) {
    console.log(data)
  })
}
catch(err) {
  datapath = "../../"
  d3.csv(datapath + "results_features_good/RFE_two_subtypes_freq.csv", function(data) {
    console.log(data)})
}
d3.csv(datapath+ "results_features_good/RFE_two_subtypes_feat.csv", function(data) {
var feature_array = [];
var type_array = [];


data.forEach(function(entry) {
    if (!feature_array.includes(entry.features)){
        feature_array.push(entry.features)
    };

    if (!type_array.includes(entry.type)){
    type_array.push(entry.type)
  };
});
var svg = d3.select("#my_dataviz2")
.append("svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
.append("g")
  .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")");

// Labels of row and columns
var myGroups = feature_array
var myVars = type_array

// Build X scales and axis:
var x = d3.scaleBand()
  .range([ 0, width ])
  .domain(myGroups)
  .padding(0.01);
svg.append("g")
  .attr("transform", "translate(0," + height + ")")
  .call(d3.axisBottom(x))
  .selectAll("text")
      .style("text-anchor", "end")
      .style("font-size", "5px")
      .attr("dx", "-.8em")
      .attr("dy", ".15em")
      .attr("transform", "rotate(-65)");

// Build X scales and axis:
var y = d3.scaleBand()
  .range([ height, 0 ])
  .domain(myVars)
  .padding(0.01);
svg.append("g")
  .call(d3.axisLeft(y));


// Build color scale
var myColor = d3.scaleLinear()
  .range(["red", "green"])
  .domain([0,1]);
  var tooltip = d3.select("#my_dataviz2")
      .append("div")
      .attr("class", "tooltip")
      .style("background-color", "white")
      .style("border", "solid")
      .style("border-width", "2px")
      .style("border-radius", "5px")
      .style("padding", "5px")
    // Three function that change the tooltip when user hover / move / leave a cell
    var mouseover = function(d) {
      tooltip
      d3.select(this)
        .style("stroke", "black")
    }
    var mousemove = function(d) {
      tooltip
        .html("Feature: " + d.features + " </b>" + " Frequency: " + d.freqs + "</b>" + " Accuracy: " + d.accuracy )
        .style("left", (d3.mouse(this[0] + 70) + "px"))
        .style("top", (d3.mouse(this)[1]) + "px")
    }
    var mouseleave = function(d) {
      tooltip
      d3.select(this)
        .style("stroke", "none")
    }

  svg.selectAll()
      .data(data)
      .enter()
      .append("rect")
      .attr("x", function(d) { return x(d.features) })
      .attr("y", function(d) { return y(d.type) })
      .attr("width", x.bandwidth() )
      .attr("height", y.bandwidth() )
      .style("fill", function(d) { return myColor(+d.accuracy)} )
          .on("mouseover", mouseover)
          .on("mousemove", mousemove)
          .on("mouseleave", mouseleave);



})
