"use client";

import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import clsx from "clsx";

const API_ENV = process.env.NEXT_PUBLIC_API_URL;
const API = API_ENV && !API_ENV.includes("localhost")
  ? API_ENV
  : process.env.NODE_ENV === "development"
    ? "http://localhost:8000"
    : "/api";

type Project = {
  id: string;
  name: string;
  allow_robots: boolean;
  created_at: string;
};

type Source = {
  url: string;
  mode: "single" | "index" | "sitemap";
  max_pages: number;
};

type Tag = {
  name: string;
  description: string;
  seed_keywords?: string[];
  example_questions?: string[];
};

type TagWeight = {
  name: string;
  weight: number;
};

type SyllabusLecture = {
  lecture_number: number;
  title: string;
  summary: string;
  learning_objectives: string[];
  key_terms: string[];
  recommended_sources: { url: string; rationale: string }[];
  estimated_reading_time_minutes: number;
  focus_tags: TagWeight[];
};

type Job = {
  id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  progress: number;
  message?: string;
  job_type: string;
};

type ScrapeSummary = {
  ok: number;
  skipped: number;
  failed: number;
  words: number;
  discovered?: number;
  updated_at: string;
};

type ScrapeError = {
  url: string;
  error: string;
};

async function apiFetch<T>(
  path: string,
  options?: RequestInit,
  timeoutMs: number = 20000
): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  let res: Response;
  try {
    res = await fetch(`${API}${path}`, {
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      ...options,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error("Request timed out. The API took too long to respond.");
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

function formatTimestamp(date = new Date()) {
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

export default function Page() {
  const [clientTime, setClientTime] = useState<string>("--:--:--");
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<string | null>(null);
  const [projectName, setProjectName] = useState("");

  const [sourceText, setSourceText] = useState("");
  const [sourceMode, setSourceMode] = useState<Source["mode"]>("single");
  const [sourceMaxPages, setSourceMaxPages] = useState(25);

  const [tags, setTags] = useState<Tag[]>([]);
  const [tagWeights, setTagWeights] = useState<Record<string, number>>({});

  const [syllabus, setSyllabus] = useState<SyllabusLecture[]>([]);

  const [lectureFiles, setLectureFiles] = useState<string[]>([]);
  const [lectureContent, setLectureContent] = useState<string>("");

  const [essayTopic, setEssayTopic] = useState("");
  const [essayFiles, setEssayFiles] = useState<string[]>([]);
  const [essayContent, setEssayContent] = useState<string>("");
  const [lectureChunkedEnabled, setLectureChunkedEnabled] = useState(true);
  const [lectureChunkParts, setLectureChunkParts] = useState(3);
  const [lectureChunkOverhead, setLectureChunkOverhead] = useState(900);
  const [syllabusLectureCount, setSyllabusLectureCount] = useState(30);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [jobStart, setJobStart] = useState<number | null>(null);
  const [statusMessage, setStatusMessage] = useState<string>("");
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [scrapeSummary, setScrapeSummary] = useState<ScrapeSummary | null>(null);
  const [scrapeErrors, setScrapeErrors] = useState<ScrapeError[]>([]);
  const isBusy = Boolean(busyAction) || job?.status === "running";
  const [pipelineStage, setPipelineStage] = useState<
    "idle" | "scrape" | "repo" | "tags" | "syllabus" | "done"
  >("idle");
  const [pipelineError, setPipelineError] = useState<string>("");
  const [currentStep, setCurrentStep] = useState(0);
  const showOverlay = isBusy || job?.status === "running";
  const overlayMessage = busyAction || (job ? `${job.job_type} ${job.status}` : "");
  const overlayProgress = job ? Math.min(100, Math.max(0, Math.round(job.progress * 100))) : 0;
  const overlayEta =
    job && jobStart && job.progress > 0
      ? Math.max(
          0,
          Math.round(
            ((Date.now() - jobStart) / 1000) * (1 / job.progress - 1) / 60
          )
        )
      : null;

  useEffect(() => {
    apiFetch<Project[]>("/projects").then(setProjects).catch(() => null);
    const stored = window.localStorage.getItem("ethed-project");
    if (stored) {
      setSelectedProject(stored);
    }
  }, []);

  useEffect(() => {
    setClientTime(formatTimestamp());
    const interval = setInterval(() => {
      setClientTime(formatTimestamp());
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!selectedProject) return;
    window.localStorage.setItem("ethed-project", selectedProject);
    setPipelineStage("idle");
    setPipelineError("");
    apiFetch<{ sources: Source[] }>(`/projects/${selectedProject}/sources`)
      .then((data) => {
        const urls = data.sources.map((s) => s.url).join("\n");
        setSourceText(urls);
      })
      .catch(() => null);

    apiFetch<{ tags: Tag[] }>(`/projects/${selectedProject}/tags`)
      .then((data) => {
        setTags(data.tags);
        const weights: Record<string, number> = {};
        data.tags.forEach((tag) => (weights[tag.name] = 1));
        setTagWeights(weights);
      })
      .catch(() => null);

    apiFetch<{ lectures: SyllabusLecture[] }>(
      `/projects/${selectedProject}/syllabus/draft`
    )
      .then((data) => setSyllabus(data.lectures))
      .catch(() => null);

    apiFetch<{ files: string[] }>(`/projects/${selectedProject}/lectures`)
      .then((data) => setLectureFiles(data.files))
      .catch(() => null);

    apiFetch<{ files: string[] }>(`/projects/${selectedProject}/essays`)
      .then((data) => setEssayFiles(data.files))
      .catch(() => null);

    apiFetch<ScrapeSummary>(`/projects/${selectedProject}/scrape/summary`)
      .then((data) => setScrapeSummary(data))
      .catch(() => setScrapeSummary(null));

    apiFetch<{ errors: ScrapeError[] }>(
      `/projects/${selectedProject}/scrape/errors?limit=5`
    )
      .then((data) => setScrapeErrors(data.errors))
      .catch(() => setScrapeErrors([]));
  }, [selectedProject]);

  useEffect(() => {
    if (!jobId) return;
    const interval = setInterval(async () => {
      try {
        const data = await apiFetch<Job>(`/jobs/${jobId}`, undefined, 60000);
        setJob(data);
        if (data.status === "succeeded" || data.status === "failed") {
          clearInterval(interval);
          setJobId(null);
          setBusyAction(null);
          setStatusMessage(data.message || "");
          setErrorMessage(data.status === "failed" ? data.message || "" : "");
          if (data.job_type === "scrape" && selectedProject) {
            apiFetch<ScrapeSummary>(
              `/projects/${selectedProject}/scrape/summary`
            )
              .then((summary) => setScrapeSummary(summary))
              .catch(() => setScrapeSummary(null));
            apiFetch<{ errors: ScrapeError[] }>(
              `/projects/${selectedProject}/scrape/errors?limit=5`
            )
              .then((summary) => setScrapeErrors(summary.errors))
              .catch(() => setScrapeErrors([]));
          }
        }
      } catch {
        clearInterval(interval);
        setBusyAction(null);
      }
    }, 1500);
    return () => clearInterval(interval);
  }, [jobId, selectedProject]);

  const weightsList = useMemo<TagWeight[]>(
    () =>
      tags.map((tag) => ({
        name: tag.name,
        weight: tagWeights[tag.name] ?? 1,
      })),
    [tags, tagWeights]
  );

  const sourceCount = useMemo(
    () =>
      sourceText
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean).length,
    [sourceText]
  );

  const requireProject = () => {
    if (!selectedProject) {
      setStatusMessage("Select a project first.");
      return false;
    }
    return true;
  };

  const validateSources = (text: string) => {
    const lines = text
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    const invalid: string[] = [];
    const valid: string[] = [];
    for (const line of lines) {
      const cleaned = line.replace(/[),.;]+$/g, "");
      try {
        const url = new URL(cleaned);
        if (!url.protocol.startsWith("http")) {
          invalid.push(line);
        } else {
          valid.push(cleaned);
        }
      } catch {
        invalid.push(line);
      }
    }
    return { valid, invalid };
  };

  const runAction = async (label: string, action: () => Promise<void>) => {
    setBusyAction(label);
    setStatusMessage(label);
    setErrorMessage("");
    try {
      await action();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Request failed. Check the API server.";
      setErrorMessage(message);
      setStatusMessage("Action failed.");
      setBusyAction(null);
      return;
    }
    setBusyAction(null);
  };

  const waitForJob = async (jobToTrack: Job) => {
    setJobId(jobToTrack.id);
    setJob(jobToTrack);
    setJobStart(Date.now());
    return new Promise<Job>((resolve, reject) => {
      const interval = setInterval(async () => {
        try {
          const data = await apiFetch<Job>(`/jobs/${jobToTrack.id}`, undefined, 60000);
          setJob(data);
          if (data.status === "succeeded") {
            clearInterval(interval);
            setJobId(null);
            setJobStart(null);
            resolve(data);
          }
          if (data.status === "failed") {
            clearInterval(interval);
            setJobId(null);
            setJobStart(null);
            reject(new Error(data.message || "Job failed"));
          }
        } catch (err) {
          clearInterval(interval);
          setJobId(null);
          setJobStart(null);
          reject(err);
        }
      }, 1500);
    });
  };

  const runAutopilot = async () => {
    if (!requireProject()) return;
    const { valid, invalid } = validateSources(sourceText);
    if (invalid.length) {
      setErrorMessage(`Invalid source URLs:\\n${invalid.join("\\n")}`);
      setStatusMessage("Action failed.");
      return;
    }
    setPipelineError("");
    setBusyAction("Running pipeline...");
    setStatusMessage("Running pipeline...");
    setErrorMessage("");
    try {
      const sources = valid.map((url) => ({
        url,
        mode: sourceMode,
        max_pages: sourceMaxPages,
      }));
      await apiFetch(`/projects/${selectedProject}/sources`, {
        method: "POST",
        body: JSON.stringify({ sources }),
      });

      setPipelineStage("scrape");
      const scrapeJob = await apiFetch<Job>(
        `/projects/${selectedProject}/scrape`,
        {
          method: "POST",
        }
      );
      await waitForJob(scrapeJob);

      setPipelineStage("repo");
      const repoJob = await apiFetch<Job>(
        `/projects/${selectedProject}/repo/build`,
        { method: "POST" }
      );
      await waitForJob(repoJob);

      setPipelineStage("tags");
      const tagsData = await apiFetch<{ tags: Tag[] }>(
        `/projects/${selectedProject}/tags/generate`,
        { method: "POST", body: JSON.stringify({ sample_size: 20 }) },
        60000
      );
      setTags(tagsData.tags);
      const weights: Record<string, number> = {};
      tagsData.tags.forEach((tag) => (weights[tag.name] = 1));
      setTagWeights(weights);
      const localWeightsList = tagsData.tags.map((tag) => ({
        name: tag.name,
        weight: 1,
      }));

      setPipelineStage("syllabus");
      const syllabusJob = await apiFetch<Job>(
        `/projects/${selectedProject}/syllabus/generate-job`,
        {
          method: "POST",
          body: JSON.stringify({
            tag_weights: localWeightsList,
            lecture_count: syllabusLectureCount,
          }),
        },
        60000
      );
      await waitForJob(syllabusJob);
      const syllabusData = await apiFetch<{ lectures: SyllabusLecture[] }>(
        `/projects/${selectedProject}/syllabus/draft`
      );
      setSyllabus(syllabusData.lectures);

      setPipelineStage("done");
      setBusyAction(null);
      setStatusMessage("Pipeline complete.");
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Pipeline failed. Check the API.";
      setPipelineError(message);
      setErrorMessage(message);
      setStatusMessage("Pipeline failed.");
      setBusyAction(null);
    }
  };

  const handleCreateProject = async () => {
    if (!projectName.trim()) {
      setStatusMessage("Enter a project name.");
      return;
    }
    await runAction("Creating project...", async () => {
      const project = await apiFetch<Project>("/projects", {
        method: "POST",
        body: JSON.stringify({ name: projectName }),
      });
      setProjects((prev) => [...prev, project]);
      setSelectedProject(project.id);
      setProjectName("");
      setBusyAction(null);
    });
  };

  const handleDeleteProject = async (projectId: string) => {
    if (!window.confirm("Delete this project and all its data?")) return;
    await runAction("Deleting project...", async () => {
      await apiFetch(`/projects/${projectId}`, { method: "DELETE" });
      setProjects((prev) => prev.filter((project) => project.id !== projectId));
      if (selectedProject === projectId) {
        setSelectedProject(null);
        setSourceText("");
        setTags([]);
        setTagWeights({});
        setSyllabus([]);
        setLectureFiles([]);
        setLectureContent("");
        setEssayFiles([]);
        setEssayContent("");
      }
      setBusyAction(null);
      setStatusMessage("Project deleted.");
    });
  };

  const handleClearAllProjects = async () => {
    if (!window.confirm("Clear all projects and all stored data?")) return;
    await runAction("Clearing all projects...", async () => {
      await apiFetch("/projects", { method: "DELETE" });
      setProjects([]);
      setSelectedProject(null);
      setSourceText("");
      setTags([]);
      setTagWeights({});
      setSyllabus([]);
      setLectureFiles([]);
      setLectureContent("");
      setEssayFiles([]);
      setEssayContent("");
      setBusyAction(null);
      setStatusMessage("All projects cleared.");
    });
  };

  const handleSaveSources = async () => {
    if (!requireProject()) return;
    const { valid, invalid } = validateSources(sourceText);
    if (invalid.length) {
      setErrorMessage(`Invalid source URLs:\\n${invalid.join("\\n")}`);
      setStatusMessage("Action failed.");
      return;
    }
    const sources = valid.map((url) => ({
      url,
      mode: sourceMode,
      max_pages: sourceMaxPages,
    }));
    await runAction("Saving sources...", async () => {
      await apiFetch(`/projects/${selectedProject}/sources`, {
        method: "POST",
        body: JSON.stringify({ sources }),
      });
      setStatusMessage("Sources saved.");
      setBusyAction(null);
    });
  };

  const handleScrape = async () => {
    if (!requireProject()) return;
    await runAction("Starting scrape...", async () => {
      const job = await apiFetch<Job>(`/projects/${selectedProject}/scrape`, {
        method: "POST",
      });
      setJobStart(Date.now());
      setJobId(job.id);
      setJob(job);
    });
  };

  const handleBuildRepo = async () => {
    if (!requireProject()) return;
    await runAction("Building repo...", async () => {
      const job = await apiFetch<Job>(`/projects/${selectedProject}/repo/build`, {
        method: "POST",
      });
      setJobStart(Date.now());
      setJobId(job.id);
      setJob(job);
    });
  };

  const handleGenerateTags = async () => {
    if (!requireProject()) return;
    await runAction("Generating tags...", async () => {
      const { valid } = validateSources(sourceText);
      const sources = valid.map((url) => ({
        url,
        mode: sourceMode,
        max_pages: sourceMaxPages,
      }));
      const data = await apiFetch<{ tags: Tag[] }>(
        `/projects/${selectedProject}/tags/generate`,
        { method: "POST", body: JSON.stringify({ sample_size: 20, sources }) },
        60000
      );
      setTags(data.tags);
      const weights: Record<string, number> = {};
      data.tags.forEach((tag) => (weights[tag.name] = 1));
      setTagWeights(weights);
      setBusyAction(null);
    });
  };

  const handleSaveTags = async () => {
    if (!requireProject()) return;
    await runAction("Saving tags...", async () => {
      await apiFetch(`/projects/${selectedProject}/tags`, {
        method: "PUT",
        body: JSON.stringify({ tags }),
      });
      setStatusMessage("Tags saved.");
      setBusyAction(null);
    });
  };

  const handleGenerateSyllabus = async () => {
    if (!requireProject()) return;
    await runAction("Generating syllabus...", async () => {
      const job = await apiFetch<Job>(
        `/projects/${selectedProject}/syllabus/generate-job`,
        {
          method: "POST",
          body: JSON.stringify({
            tag_weights: weightsList,
            lecture_count: syllabusLectureCount,
          }),
        },
        60000
      );
      await waitForJob(job);
      const data = await apiFetch<{ lectures: SyllabusLecture[] }>(
        `/projects/${selectedProject}/syllabus/draft`
      );
      setSyllabus(data.lectures);
      setBusyAction(null);
    });
  };

  const handleApproveSyllabus = async () => {
    if (!requireProject()) return;
    await runAction("Approving syllabus...", async () => {
      await apiFetch(`/projects/${selectedProject}/syllabus/approve`, {
        method: "POST",
      });
      setStatusMessage("Syllabus approved.");
      setBusyAction(null);
    });
  };


  const handlePurgeLectures = async () => {
    if (!requireProject()) return;
    await runAction("Purging lectures...", async () => {
      await apiFetch<{ deleted: number }>(`/projects/${selectedProject}/lectures`, { method: "DELETE" });
      setLectureFiles([]);
      setLectureContent("");
    });
  };

  const handleGenerateLectures = async () => {
    if (!requireProject()) return;
    await runAction("Generating lectures...", async () => {
      const job = await apiFetch<Job>(
        `/projects/${selectedProject}/lectures/generate`,
        {
          method: "POST",
          body: JSON.stringify({
            tag_weights: weightsList,
            lecture_chunked_enabled: lectureChunkedEnabled,
            lecture_chunk_parts: lectureChunkParts,
            lecture_chunk_overhead_words: lectureChunkOverhead,
          }),
        },
        60000
      );
      setJobStart(Date.now());
      setJobId(job.id);
      setJob(job);
    });
  };

  const handleDeleteCache = async () => {
    if (!requireProject()) return;
    await runAction("Deleting cache...", async () => {
      await apiFetch(`/projects/${selectedProject}/cache`, { method: "DELETE" });
      setStatusMessage("Cache deleted.");
      setBusyAction(null);
    });
  };

  const handleDeleteRepo = async () => {
    if (!requireProject()) return;
    await runAction("Deleting repo...", async () => {
      await apiFetch(`/projects/${selectedProject}/repo`, { method: "DELETE" });
      setStatusMessage("Repo cleared.");
      setScrapeSummary(null);
      setScrapeErrors([]);
      setSyllabus([]);
      setLectureFiles([]);
      setLectureContent("");
      setBusyAction(null);
    });
  };

  const handleLoadLecture = async (file: string) => {
    if (!requireProject()) return;
    await runAction("Loading lecture...", async () => {
      const number = Number(file.split("-")[0]);
      const data = await apiFetch<{ content: string }>(
        `/projects/${selectedProject}/lectures/${number}`
      );
      setLectureContent(data.content);
      setBusyAction(null);
    });
  };

  const handleRefreshLectures = async () => {
    if (!requireProject()) return;
    await runAction("Refreshing lectures...", async () => {
      const data = await apiFetch<{ files: string[] }>(
        `/projects/${selectedProject}/lectures`
      );
      setLectureFiles(data.files);
      setBusyAction(null);
    });
  };

  const handleGenerateEssay = async () => {
    if (!requireProject()) return;
    if (!essayTopic.trim()) {
      setStatusMessage("Enter an essay topic.");
      return;
    }
    await runAction("Generating essay...", async () => {
      await apiFetch(`/projects/${selectedProject}/essays/generate`, {
        method: "POST",
        body: JSON.stringify({
          topic: essayTopic,
          tag_weights: weightsList,
          length: "medium",
        }),
      });
      const data = await apiFetch<{ files: string[] }>(
        `/projects/${selectedProject}/essays`
      );
      setEssayFiles(data.files);
      setEssayTopic("");
      setBusyAction(null);
    });
  };

  const handleLoadEssay = async (file: string) => {
    if (!requireProject()) return;
    await runAction("Loading essay...", async () => {
      const data = await apiFetch<{ content: string }>(
        `/projects/${selectedProject}/essays/${file}`
      );
      setEssayContent(data.content);
      setBusyAction(null);
    });
  };

  const logRows = useMemo(() => {
    const rows: { id: string; message: string; status: string }[] = [];
    if (errorMessage) {
      rows.push({ id: "ERROR", message: errorMessage, status: "ERROR" });
    }
    if (statusMessage) {
      rows.push({ id: "STATUS", message: statusMessage, status: "OK" });
    }
    if (job) {
      rows.push({
        id: job.job_type.toUpperCase(),
        message: `Job ${job.job_type} is ${job.status} (${Math.round(
          job.progress * 100
        )}%)`,
        status: job.status === "failed" ? "ERROR" : "OK",
      });
    }
    if (scrapeSummary) {
      const ok = typeof scrapeSummary.ok === "number" ? scrapeSummary.ok : 0;
      const failed = typeof scrapeSummary.failed === "number" ? scrapeSummary.failed : 0;
      const skipped = typeof scrapeSummary.skipped === "number" ? scrapeSummary.skipped : 0;
      const words = typeof scrapeSummary.words === "number" ? scrapeSummary.words : 0;
      const discovered =
        typeof scrapeSummary.discovered === "number"
          ? scrapeSummary.discovered
          : ok + failed + skipped;
      rows.push({
        id: "SCR-SUM",
        message: `Scrape summary: ${ok} ok, ${failed} failed, ${skipped} skipped, ${words.toLocaleString()} words, ${discovered} urls`,
        status: failed > 0 ? "WARN" : "OK",
      });
    }
    scrapeErrors.forEach((err, idx) => {
      rows.push({
        id: `SCR-ERR-${idx + 1}`,
        message: `${err.url} — ${err.error}`,
        status: "ERROR",
      });
    });
    if (pipelineError) {
      rows.push({ id: "PIPE", message: pipelineError, status: "ERROR" });
    }
    if (!rows.length) {
      rows.push({ id: "SYS", message: "Idle. Ready for input.", status: "OK" });
    }
    return rows;
  }, [job, scrapeSummary, scrapeErrors, pipelineError, errorMessage, statusMessage]);

  const steps = [
    { id: 1, title: "Name Project", desc: "Create or select the project workspace for this run." },
    { id: 2, title: "Upload Links", desc: "Add and validate sources for scraping and indexing." },
    { id: 3, title: "Scrape & Build Repo", desc: "Fetch pages and build the local chunk repository." },
    { id: 4, title: "Generate Tags", desc: "Create a topic taxonomy and set weights." },
    { id: 5, title: "Generate Syllabus", desc: "Produce the lecture plan, then approve." },
    { id: 6, title: "Generate Lectures", desc: "Run long-form lecture generation and export." },
    { id: 7, title: "Essays & Review", desc: "Optional essays and content previews." },
  ];

  const pipelineNodes = [
    {
      key: "sources",
      title: "Sources",
      stat: sourceCount ? `${sourceCount} URLs` : "--",
      sub: selectedProject ? "Configured" : "No project",
    },
    {
      key: "scrape",
      title: "Scrape",
      stat:
        pipelineStage === "scrape" || job?.job_type === "scrape"
          ? `${Math.round((job?.progress || 0) * 100)}%`
          : scrapeSummary
          ? `${scrapeSummary.ok}`
          : "--",
      sub: scrapeSummary ? `${scrapeSummary.words} words` : "Awaiting",
    },
    {
      key: "repo",
      title: "Repo",
      stat: pipelineStage === "repo" ? "Building" : "Ready",
      sub: "chunks.jsonl",
    },
    {
      key: "tags",
      title: "Tags",
      stat: tags.length ? `${tags.length}` : "--",
      sub: "12 target",
    },
    {
      key: "syllabus",
      title: "Syllabus",
      stat: syllabus.length ? `${syllabus.length}` : "--",
      sub: "30 lectures",
    },
  ];

  const currentNode = pipelineStage === "idle" ? "sources" : pipelineStage;
  const pipelineIndex = Math.max(0, pipelineNodes.findIndex((node) => node.key === currentNode));
  const pipelineProgress = Math.round(((pipelineIndex + 1) / pipelineNodes.length) * 100);
  const nextStep = () => setCurrentStep((prev) => Math.min(prev + 1, steps.length - 1));
  const prevStep = () => setCurrentStep((prev) => Math.max(prev - 1, 0));
  const stepData = steps[currentStep];
  const stepProgress = Math.round(((currentStep + 1) / steps.length) * 100);

  return (
    <div className="app-shell">
      {showOverlay && (
        <div className="overlay">
          <div className="overlay-card">
            <div className="uppercase bold" style={{ letterSpacing: "2px" }}>
              Processing
            </div>
            <div style={{ marginTop: 8, fontSize: 20, fontWeight: 700 }}>
              {overlayMessage || "Working..."}
            </div>
            <div className="overlay-progress">
              <div style={{ width: `${overlayProgress}%` }} />
            </div>
            <div
              style={{
                marginTop: 8,
                display: "flex",
                justifyContent: "space-between",
                fontSize: 11,
                textTransform: "uppercase",
                letterSpacing: "1px",
              }}
            >
              <span>{overlayProgress}%</span>
              <span>{overlayEta !== null ? `ETA ~ ${overlayEta} min` : "Estimating"}</span>
            </div>
            {errorMessage && (
              <div style={{ marginTop: 12, fontSize: 12, color: "#b10000" }}>
                {errorMessage}
              </div>
            )}
          </div>
        </div>
      )}
      <aside className="context-panel">
        <div>
          <div className="brand">SelfStudy</div>
          <div className="step-indicator">
            <div className="step-number">{String(stepData.id).padStart(2, "0")}</div>
            <div className="step-title">{stepData.title}</div>
            <div className="step-desc">{stepData.desc}</div>
          </div>
        </div>
        <div className="nav-controls">
          <button className="btn-nav" onClick={prevStep} disabled={currentStep === 0}>
            ←
          </button>
          <button className="btn-nav" onClick={nextStep} disabled={currentStep === steps.length - 1}>
            →
          </button>
        </div>
      </aside>

      <div className="axis-line">
        <div className="axis-progress" style={{ "--axis-progress": `${stepProgress}%` } as Record<string, string>} />
      </div>

      <main className="main-content">
        <header className="hud">
          <div className="hud-item">
            <span className="hud-label">Active Project</span>
            <span className="hud-value">{selectedProject || "None"}</span>
          </div>
          <div className="hud-item">
            <span className="hud-label">Status</span>
            <span className="hud-value">{isBusy ? "Running" : "Idle"}</span>
          </div>
          <div className="hud-item mono">
            <span className="hud-label">ID</span>
            <span className="hud-value">{selectedProject ? selectedProject.slice(0, 6) : "----"}</span>
          </div>
        </header>

        <div className="hud-log">
          <div className="hud-log-title uppercase">Live Logs</div>
          <div className="hud-log-body">
            {logRows.map((row, idx) => (
              <div key={`${row.id}-${idx}`} className={clsx("hud-line", { active: idx === 0 })}>
                <span>{row.id}</span>
                <span>{row.message}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="stage-container">
          <div className="stage-viewport">
            <div
              className="stage-track"
              style={{
                "--track-offset": `${(-100 * currentStep) / steps.length}%`,
                "--panel-count": String(steps.length),
              } as Record<string, string>}
            >
            <section className="panel">
              <div className="panel-inner">
                <div className="panel-header">Project</div>
                <div className="form-group">
                  <label className="form-label">Create new project</label>
                  <input
                    className="form-input"
                    placeholder="Ethereum foundations"
                    value={projectName}
                    onChange={(e) => setProjectName(e.target.value)}
                  />
                  <button className="btn-primary" onClick={handleCreateProject} disabled={isBusy}>
                    Create project
                  </button>
                </div>
                <div className="form-group">
                  <label className="form-label">Select project</label>
                  <div className="panel-list">
                    {projects.map((project) => (
                      <div key={project.id} className="panel-row">
                        <button
                          className="btn-ghost"
                          style={{
                            textAlign: "left",
                            background: selectedProject === project.id ? "#1a1a1a" : "transparent",
                            color: selectedProject === project.id ? "#ebece8" : "#1a1a1a",
                          }}
                          onClick={() => setSelectedProject(project.id)}
                        >
                          {project.name}
                          <div className="mono" style={{ fontSize: 10, opacity: 0.6 }}>
                            {project.id}
                          </div>
                        </button>
                        <button
                          className="btn-ghost"
                          style={{ borderColor: "#b10000", color: "#b10000" }}
                          onClick={() => handleDeleteProject(project.id)}
                        >
                          Delete
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
                <button
                  className="btn-secondary"
                  style={{ borderColor: "#b10000", color: "#b10000" }}
                  onClick={handleClearAllProjects}
                  disabled={isBusy}
                >
                  Clear all project data
                </button>
              </div>
            </section>

            <section className="panel">
              <div className="panel-inner">
                <div className="panel-header">Sources</div>
                <div className="form-group">
                  <label className="form-label">URLs (one per line)</label>
                  <textarea
                    className="form-textarea"
                    placeholder="https://..."
                    value={sourceText}
                    onChange={(e) => setSourceText(e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Discovery mode</label>
                  <select
                    className="form-select"
                    value={sourceMode}
                    onChange={(e) => setSourceMode(e.target.value as Source["mode"])}
                  >
                    <option value="single">Single</option>
                    <option value="index">Index</option>
                    <option value="sitemap">Sitemap</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Max pages</label>
                  <input
                    className="form-input"
                    type="number"
                    min={1}
                    value={sourceMaxPages}
                    onChange={(e) => setSourceMaxPages(Number(e.target.value))}
                  />
                </div>
                <div className="panel-actions">
                  <button className="btn-primary" onClick={handleSaveSources} disabled={isBusy}>
                    Save sources
                  </button>
                  <button className="btn-secondary" onClick={handleScrape} disabled={isBusy}>
                    Start scrape
                  </button>
                </div>
              </div>
            </section>

            <section className="panel">
              <div className="panel-inner">
                <div className="panel-header">Scrape & Repo</div>
                <div className="console-output">
                  {logRows.map((row, idx) => (
                    <div key={`${row.id}-${idx}`} className={clsx("console-line", { active: idx === 0 })}>
                      &gt; {row.message}
                    </div>
                  ))}
                </div>
                <div className="panel-actions">
                  <button className="btn-primary" onClick={handleBuildRepo} disabled={isBusy}>
                    Build repo
                  </button>
                  <button className="btn-secondary" onClick={runAutopilot} disabled={isBusy}>
                    Run pipeline
                  </button>
                  <button className="btn-ghost" onClick={handleDeleteCache} disabled={isBusy}>
                    Delete cache
                  </button>
                  <button className="btn-ghost" onClick={handleDeleteRepo} disabled={isBusy}>
                    Delete repo
                  </button>
                </div>
              </div>
            </section>

            <section className="panel panel-scroll">
              <div className="panel-inner">
                <div className="panel-header">Tags & Weights</div>
                <div className="panel-actions">
                  <button className="btn-primary" onClick={handleGenerateTags} disabled={isBusy}>
                    Generate 12 tags
                  </button>
                  <button className="btn-secondary" onClick={handleSaveTags} disabled={isBusy}>
                    Save tags
                  </button>
                </div>
                <div className="panel-list">
                  {tags.map((tag, idx) => (
                    <div key={idx} className="panel-card">
                      <input
                        className="form-input"
                        value={tag.name}
                        onChange={(e) => {
                          const next = [...tags];
                          next[idx] = { ...tag, name: e.target.value };
                          setTags(next);
                        }}
                      />
                      <textarea
                        className="form-textarea"
                        value={tag.description}
                        onChange={(e) => {
                          const next = [...tags];
                          next[idx] = { ...tag, description: e.target.value };
                          setTags(next);
                        }}
                      />
                      <div className="weight-row">
                        <span className="weight-label">Weight</span>
                        <input
                          className="weight-range"
                          type="range"
                          min={0}
                          max={5}
                          step={0.5}
                          value={tagWeights[tag.name] ?? 1}
                          onChange={(e) =>
                            setTagWeights((prev) => ({
                              ...prev,
                              [tag.name]: Number(e.target.value),
                            }))
                          }
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </section>

            <section className="panel">
              <div className="panel-inner">
                <div className="panel-header">Syllabus</div>
                <div className="form-group">
                  <label className="form-label">Lecture count</label>
                  <input
                    className="form-input"
                    type="number"
                    min={3}
                    max={60}
                    value={syllabusLectureCount}
                    onChange={(e) => setSyllabusLectureCount(Number(e.target.value))}
                  />
                </div>
                <div className="panel-actions">
                  <button className="btn-primary" onClick={handleGenerateSyllabus} disabled={isBusy}>
                    Generate syllabus
                  </button>
                  <button className="btn-secondary" onClick={handleApproveSyllabus} disabled={isBusy}>
                    Approve syllabus
                  </button>
                </div>
                <div className="panel-list">
                  {syllabus.length === 0 && <div className="panel-card">No draft yet.</div>}
                  {syllabus.map((lecture) => (
                    <div key={lecture.lecture_number} className="panel-card">
                      <div className="uppercase" style={{ fontSize: 10, letterSpacing: 1 }}>
                        Lecture {lecture.lecture_number}
                      </div>
                      <div style={{ fontWeight: 700 }}>{lecture.title}</div>
                      <div style={{ fontSize: 12, marginTop: 4, color: "#444" }}>
                        {lecture.summary}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </section>

            <section className="panel">
              <div className="panel-inner">
                <div className="panel-header">Lectures</div>
                <div className="form-group">
                  <label className="form-label">Chunked generation</label>
                  <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <input
                      type="checkbox"
                      checked={lectureChunkedEnabled}
                      onChange={(e) => setLectureChunkedEnabled(e.target.checked)}
                    />
                    Enable chunked lecture generation
                  </label>
                </div>
                <div className="form-group">
                  <label className="form-label">Chunk parts</label>
                  <input
                    className="form-input"
                    type="number"
                    min={1}
                    max={6}
                    value={lectureChunkParts}
                    onChange={(e) => setLectureChunkParts(Number(e.target.value))}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Overhead words</label>
                  <input
                    className="form-input"
                    type="number"
                    min={400}
                    max={2000}
                    value={lectureChunkOverhead}
                    onChange={(e) => setLectureChunkOverhead(Number(e.target.value))}
                  />
                </div>
                <div className="panel-actions">
                  <button className="btn-primary" onClick={handleGenerateLectures} disabled={isBusy}>
                    Generate lectures
                  </button>
                  <button className="btn-secondary" onClick={handleRefreshLectures} disabled={isBusy}>
                    Refresh list
                  </button>
                  {selectedProject && (
                    <a
                      className="btn-secondary"
                      href={`${API}/projects/${selectedProject}/downloads/lectures.zip`}
                    >
                      Download zip
                    </a>
                  )}
                </div>
                <div className="panel-list">
                  {lectureFiles.map((file) => (
                    <button key={file} className="btn-ghost" onClick={() => handleLoadLecture(file)}>
                      {file}
                    </button>
                  ))}
                </div>
              </div>
            </section>

            <section className="panel">
              <div className="panel-inner">
                <div className="panel-header">Essays & Preview</div>
                <div className="form-group">
                  <label className="form-label">Essay topic</label>
                  <input
                    className="form-input"
                    value={essayTopic}
                    onChange={(e) => setEssayTopic(e.target.value)}
                    placeholder="Emergent Ethereum institutions"
                  />
                </div>
                <button className="btn-primary" onClick={handleGenerateEssay} disabled={isBusy}>
                  Generate essay
                </button>
                <div className="panel-list">
                  {essayFiles.map((file) => (
                    <button key={file} className="btn-ghost" onClick={() => handleLoadEssay(file)}>
                      {file}
                    </button>
                  ))}
                </div>
                <div className="panel-split">
                  <div>
                    <div className="panel-subtitle">Lecture Preview</div>
                    <div className="panel-card">
                      {lectureContent ? (
                        <ReactMarkdown>{lectureContent}</ReactMarkdown>
                      ) : (
                        "Select a lecture to preview."
                      )}
                    </div>
                  </div>
                  <div>
                    <div className="panel-subtitle">Essay Preview</div>
                    <div className="panel-card">
                      {essayContent ? (
                        <ReactMarkdown>{essayContent}</ReactMarkdown>
                      ) : (
                        "Select an essay to preview."
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </section>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
